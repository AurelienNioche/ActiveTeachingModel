from django.db import models

from django.contrib.postgres.fields import ArrayField

import numpy as np
import pickle
from scipy.special import logsumexp
from tqdm import tqdm

from model.compute import compute_grid_param, \
    post_mean_sd

from model.teacher import Teacher, TeacherPerfectInfo, Psychologist, Leitner
from model.learner import ExponentialForgetting


class Simulation(models.Model):

    # Parameters ----------------------------------------
    n_item = models.IntegerField()

    n_session = models.IntegerField()
    n_iteration_per_session = models.IntegerField()
    n_iteration_between_session = models.IntegerField()

    grid_size = models.IntegerField()

    teacher_model = models.CharField(max_length=32)
    learner_model = models.CharField(max_length=32)

    param_labels = ArrayField(models.CharField(max_length=32), default=list)
    param_values = ArrayField(models.FloatField(), default=list)
    param_upper_bounds = ArrayField(models.FloatField(), default=list)
    param_lower_bounds = ArrayField(models.FloatField(), default=list)

    seed = models.IntegerField()

    # Outputs ----------------------------------------

    random_state = models.BinaryField(null=True, blank=True)

    log_post = ArrayField(models.FloatField(), default=list)

    timestamp = ArrayField(models.IntegerField(), default=list)
    hist = ArrayField(models.IntegerField(), default=list)
    success = ArrayField(models.BooleanField(), default=list)

    @property
    def post(self):

        if self._read_post is None:
            post_entries = self.post_set.all()

            post = {stat: {} for stat in ("std", "mean")}

            for pr in self.param_labels:
                post_entry_pr = post_entries.get(param_label=pr)
                post["mean"][pr] = np.array(
                    post_entry_pr.mean[:self.n_iteration])
                post["std"][pr] = np.array(
                    post_entry_pr.std[:self.n_iteration])
            self._read_post = post
        else:
            post = self._read_post
        return post

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._read_post = None

        self.hist_array = np.asarray(self.hist, dtype=int)
        self.success_array = np.asarray(self.success, dtype=bool)
        self.timestamp_array = np.asarray(self.timestamp, dtype=int)

        self.log_post_array = np.asarray(self.log_post)

        self.n_iteration = self.n_session * self.n_iteration_per_session

        self.t = None
        self.c_iter_session = None

        self.pm = None

        self.stop_event = None
        self.iterator = None
        self.grid_param = None
        self.delta = None
        self.n_pres = None
        self.n_success = None

    def setup(self, stop_event):

        self.post_array = np.array((self.n_iteration, len(self.param_values)))

        self.hist_array = np.zeros(self.n_iteration, dtype=int)
        self.success_array = np.zeros(self.n_iteration, dtype=bool)
        self.timestamp_array = np.zeros(self.n_iteration, dtype=int)

        self.post_mean_array = np.zeros((self.n_iteration,
                                         len(self.param_values)))

        self.post_sd_array = np.zeros((self.n_iteration,
                                       len(self.param_values)))

        bounds = np.zeros((len(self.param_values), 2))
        bounds[:, 0] = self.param_upper_bounds
        bounds[:, 1] = self.param_lower_bounds
        self.grid_param = compute_grid_param(bounds=bounds,
                                             grid_size=self.grid_size)
        n_param_set = len(self.grid_param)
        lp = np.ones(n_param_set)
        self.log_post_array = lp - logsumexp(lp)

        self.delta = np.zeros(self.n_item, dtype=int)
        self.n_pres = np.zeros(self.n_item, dtype=int)
        self.n_success = np.zeros(self.n_item, dtype=int)

        self.pm, self.ps = None, None

        self.c_iter_session = 0
        self.t = 0

        if stop_event:
            self.stop_event = stop_event
            self.iterator = range(self.n_iteration)
        else:
            self.iterator = tqdm(range(self.n_iteration))

        np.random.seed(self.seed)

    def prepare_extension(self, n_session, stop_event):

        previous_n_session = self.n_session
        previous_n_iteration = self.n_iteration

        print("new n session", n_session)
        print("old n session", previous_n_session)

        self.n_session = n_session
        self.n_iteration = n_session * self.n_iteration_per_session

        self.setup(stop_event=stop_event)

        self.t = (self.n_iteration_between_session
                  + self.n_iteration_per_session) * previous_n_session
        self.c_iter_session = 0

        print("old n session", previous_n_session)
        print("n iter per session", self.n_iteration_per_session)
        current_it = self.n_iteration_per_session * previous_n_session
        print("current it", current_it)
        self.iterator = range(
                current_it,
                self.n_iteration)

        # If no multiprocessing, wrap with tqdm
        if self.stop_event is None:
            self.iterator = tqdm(self.iterator)

        self.hist_array[:previous_n_iteration] = self.hist
        self.success_array[:previous_n_iteration] = self.success
        self.timestamp_array[:previous_n_iteration] = self.timestamp

        for i in range(self.n_item):

            i_pres = self.hist_array[:] == i
            n_pres_i = np.sum(i_pres)
            self.n_pres[i] = n_pres_i
            if n_pres_i:
                self.delta[i] = np.argmax(self.timestamp_array[i_pres])
                self.n_success[i] = np.sum(self.success_array[i_pres])

        self.log_post_array[:] = self.log_post

        post_entries = self.post_set.all()

        for i, pr in enumerate(self.param_labels):
            post_entry_pr = post_entries.get(param_label=pr)
            self.post_mean_array[:previous_n_iteration, i] = \
                post_entry_pr.mean
            self.post_sd_array[:previous_n_iteration, i] = \
                post_entry_pr.std

        np.random.set_state(pickle.loads(self.random_state))

    def loop(self):

        if self.learner_model == ExponentialForgetting.__name__:
            learner_model_class = ExponentialForgetting
        else:
            raise ValueError(
                f"Learner model not recognized: {self.learner_model}")

        if self.teacher_model == Leitner.__name__:
            teacher_inst = Leitner(n_item=self.n_item)
        elif self.teacher_model == Teacher.__name__:
            teacher_inst = Teacher()
        elif self.teacher_model == TeacherPerfectInfo.__name__:
            teacher_inst = TeacherPerfectInfo(param=self.param_values)
        elif self.teacher_model == Psychologist.__name__:
            teacher_inst = Psychologist()
        else:
            raise ValueError(f"Teacher model not recognized: "
                             f"{self.teacher_model}")

        for it in self.iterator:

            if self.stop_event and self.stop_event.is_set():
                return

            log_lik = learner_model_class.log_lik(
                grid_param=self.grid_param,
                delta=self.delta,
                n_pres=self.n_pres,
                n_success=self.n_success,
                hist=self.hist_array,
                timestamps=self.timestamp_array,
                t=self.t)

            if self.teacher_model == Leitner.__name__:
                i = teacher_inst.ask()

            elif self.teacher_model == Psychologist.__name__:
                i = teacher_inst.get_item(
                    learner=self.learner_model,
                    log_post=self.log_post,
                    log_lik=log_lik,
                    n_item=self.n_item,
                    grid_param=self.grid_param,
                    delta=self.delta,
                    n_pres=self.n_pres,
                    n_success=self.n_success,
                    hist=self.hist)

            else:
                i = teacher_inst.get_item(
                    n_pres=self.n_pres,
                    n_success=self.n_success,
                    param=self.pm,
                    delta=self.delta,
                    hist=self.hist,
                    learner=learner_model_class,
                    timestamps=self.timestamp,
                    t=self.t
                )

            p_recall = learner_model_class.p(
                param=self.param_values,
                delta_i=self.delta[i],
                n_pres_i=self.n_pres[i],
                n_success_i=self.n_success[i],
                i=i,
                hist=self.hist_array,
                timestamps=self.timestamp_array,
                t=self.t
            )

            response = p_recall > np.random.random()

            if self.teacher_model == Leitner:
                teacher_inst.update(item=i, response=response)

            # Update prior
            self.log_post_array += log_lik[i, :, int(response)].flatten()
            self.log_post_array -= logsumexp(self.log_post_array)

            # Compute post mean and std
            pm, psd = \
                post_mean_sd(grid_param=self.grid_param,
                             log_post=self.log_post_array)

            # Backup
            self.post_mean_array[it] = pm
            self.post_sd_array[it] = psd
            self.success_array[it] = int(response)
            self.hist_array[it] = i
            self.timestamp_array[it] = self.t

            # Make the user learn
            # Increment delta for all items
            self.delta[:] += 1
            # ...except the one for the selected design that equal one
            self.delta[i] = 1
            self.n_pres[i] += 1
            self.n_success[i] += int(response)

            self.t += 1
            self.c_iter_session += 1
            if self.c_iter_session >= self.n_iteration_per_session:
                self.delta[:] += self.n_iteration_between_session
                self.t += self.n_iteration_between_session
                self.c_iter_session = 0

    def save(self, *args, **kwargs):

        self.random_state = pickle.dumps(np.random.get_state())

        self.timestamp = list(self.timestamp_array)
        self.hist = list(self.hist_array)
        self.success = list(self.success_array)

        self.log_post = list(self.log_post_array)

        super().save()

        post_entries = self.post_set.all()
        if post_entries.count():
            for i, pr in enumerate(self.param_labels):
                e = post_entries.get(param_label=pr)
                e.std = list(self.post_sd_array[:, i])
                e.mean = list(self.post_mean_array[:, i])

        else:
            post_entries = []

            for i, pr in enumerate(self.param_labels):
                post_entries.append(
                    Post(
                        simulation=self,
                        param_label=pr,
                        std=list(self.post_sd_array[:, i]),
                        mean=list(self.post_mean_array[:, i])))

            Post.objects.bulk_create(post_entries)

    @classmethod
    def run(cls, learner_model,
            teacher_model,
            param,
            n_session=30,
            n_item=1000,
            seed=0,
            grid_size=20,
            n_iteration_per_session=150,
            n_iteration_between_session=43050,
            bounds=None,
            param_labels=None,
            stop_event=None):

        kwargs = {
            "n_item": n_item,
            "n_iteration_per_session": n_iteration_per_session,
            "n_iteration_between_session": n_iteration_between_session,
            "teacher_model": teacher_model.__name__,
            "learner_model": learner_model.__name__,
            "param_labels": list(param_labels),
            "param_values": list(param),
            "param_upper_bounds": [b[0] for b in bounds],
            "param_lower_bounds": [b[1] for b in bounds],
            "grid_size": grid_size,
            "seed": seed
        }

        sim = cls.already_existing(kwargs=kwargs, n_session=n_session)

        if sim is not None:
            print("found already existing")
            if sim.n_session == n_session:
                print("same size")
                return sim
            print("shorter")
            sim.prepare_extension(stop_event=stop_event,
                                  n_session=n_session)
        else:
            sim = Simulation(n_session=n_session, **kwargs)
            sim.setup(stop_event=stop_event)

        sim.loop()

        if stop_event is None or not stop_event.is_set():
            sim.save()

    @classmethod
    def as_shorter(cls, sim, n_session):

        n_iteration = n_session * sim.n_iteration_per_session

        sim.n_session = n_session

        sim.hist_array = sim.hist_array[:n_iteration]
        sim.success_array = sim.success_array[:n_iteration]
        sim.timestamp_array = sim.timestamp_array[:n_iteration]

        # Can not retrieve this information
        sim.log_post_array = None

        return sim

    @classmethod
    def already_existing(cls, kwargs, n_session):

        sim = None

        # Find already existing
        sim_entries = Simulation.objects.filter(**kwargs)

        # If a sim already exists with same param
        if sim_entries.count():

            # If a sim has the same number of iteration
            sim_with_same_n = sim_entries.filter(n_session=n_session)
            if sim_with_same_n.count():
                sim = sim_with_same_n[0]

            else:
                sim_entries_diff_n = \
                    sim_entries.order_by("n_session").reverse()
                sim = sim_entries_diff_n[0]

                # If a sim is longer
                if sim.n_session > n_session:
                    sim = cls.as_shorter(sim, n_session)
                elif sim.teacher_model == Leitner.__name__:
                    sim = None
                elif sim.random_state is None:
                    sim = None
        return sim


class Post(models.Model):

    simulation = models.ForeignKey(Simulation, on_delete=models.CASCADE)
    param_label = models.CharField(max_length=32)
    mean = ArrayField(models.FloatField(), default=list)
    std = ArrayField(models.FloatField(), default=list)

