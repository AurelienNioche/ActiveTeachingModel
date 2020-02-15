import os

import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm
import pickle

from model.teacher.leitner import Leitner


from model.compute import compute_grid_param, \
    post_mean_sd

from model.teacher.teacher import Teacher
from model.teacher.teacher_perfect_info import TeacherPerfectInfo
from model.teacher.psychologist import Psychologist

from simulation_data.models import Simulation, Post

EPS = np.finfo(np.float).eps
FIG_FOLDER = os.path.join("fig", "scenario")


def run_n_session(**kwargs):
    return Simulation.run(**kwargs)


def run_n_session_legacy(
        learner_model,
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

    if bounds is None:
        bounds = learner_model.bounds

    if param_labels is None:
        param_labels = learner_model.param_labels

    sim_parameters = {
            "n_item": n_item,
            "n_session": n_session,
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

    sim_entries = Simulation.objects.filter(**sim_parameters)

    if sim_entries.count():
        return sim_entries[0]

    n_iteration = n_iteration_per_session * n_session

    post_means = {pr: np.zeros(n_iteration) for pr in param_labels}
    post_sds = {pr: np.zeros(n_iteration) for pr in param_labels}

    hist = np.zeros(n_iteration, dtype=int)
    success = np.zeros(n_iteration, dtype=bool)
    timestamp = np.zeros(n_iteration)

    grid_param = compute_grid_param(bounds=bounds, grid_size=grid_size)
    n_param_set = len(grid_param)

    delta = np.zeros(n_item, dtype=int)
    n_pres = np.zeros(n_item, dtype=int)
    n_success = np.zeros(n_item, dtype=int)

    if teacher_model == Leitner:
        teacher_inst = Leitner(n_item=n_item)
    elif teacher_model == Teacher:
        teacher_inst = Teacher()
    elif teacher_model == TeacherPerfectInfo:
        teacher_inst = TeacherPerfectInfo(param=param)
    elif teacher_model == Psychologist:
        teacher_inst = Psychologist()
    else:
        raise ValueError

    lp = np.ones(n_param_set)
    log_post = lp - logsumexp(lp)

    pm, ps = None, None

    np.random.seed(seed)

    c_iter_session = 0
    t = 0

    if stop_event:
        iterator = range(n_iteration)
    else:
        iterator = tqdm(range(n_iteration))

    for it in iterator:

        if stop_event and stop_event.is_set():
            return

        log_lik = learner_model.log_lik(
            grid_param=grid_param,
            delta=delta,
            n_pres=n_pres,
            n_success=n_success,
            hist=hist,
            timestamps=timestamp,
            t=t)

        if teacher_model == Leitner:
            i = teacher_inst.ask()

        elif teacher_model == Psychologist:
            i = teacher_inst.get_item(
                learner=learner_model,
                log_post=log_post,
                log_lik=log_lik,
                n_item=n_item,
                grid_param=grid_param,
                delta=delta,
                n_pres=n_pres,
                n_success=n_success,
                hist=hist)

        else:
            i = teacher_inst.get_item(
                n_pres=n_pres,
                n_success=n_success,
                param=pm,
                delta=delta,
                hist=hist,
                learner=learner_model,
                timestamps=timestamp,
                t=t
            )

        p_recall = learner_model.p(
            param=param,
            delta_i=delta[i],
            n_pres_i=n_pres[i],
            n_success_i=n_success[i],
            i=i,
            hist=hist,
            timestamps=timestamp,
            t=t
        )

        response = p_recall > np.random.random()

        if teacher_model == Leitner:
            teacher_inst.update(item=i, response=response)

        # Update prior
        log_post += log_lik[i, :, int(response)].flatten()
        log_post -= logsumexp(log_post)

        # Make the user learn
        # Increment delta for all items
        delta[:] += 1
        # ...except the one for the selected design that equal one
        delta[i] = 1
        n_pres[i] += 1
        n_success[i] += int(response)

        # Compute post mean and std
        pm, ps = post_mean_sd(grid_param=grid_param, log_post=log_post)

        # Backup the mean/std of post dist
        for i, pr in enumerate(param_labels):
            post_means[pr][it] = pm[i]
            post_sds[pr][it] = ps[i]

        # Backup success, timestamp, hist
        success[it] = int(response)
        timestamp[it] = t
        hist[it] = i

        t += 1
        c_iter_session += 1
        if c_iter_session >= n_iteration_per_session:
            delta[:] += n_iteration_between_session
            t += n_iteration_between_session
            c_iter_session = 0

    state = pickle.dumps(np.random.get_state())

    sim_entry = Simulation.objects.create(
        timestamp=list(timestamp),
        hist=list(hist),
        success=list(success),
        log_post=list(log_post),
        random_state=state,
        **sim_parameters
    )

    post_entries = []

    for i, pr in enumerate(param_labels):

        post_entries.append(
            Post(
                simulation=sim_entry,
                param_label=pr,
                std=list(post_sds[pr]),
                mean=list(post_means[pr])))

    Post.objects.bulk_create(post_entries)

    return sim_entry
