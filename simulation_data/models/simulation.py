from django.db import models

from django.contrib.postgres.fields import ArrayField

import numpy as np


class Simulation(models.Model):

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

    timestamp = ArrayField(models.IntegerField(), default=list)
    hist = ArrayField(models.IntegerField(), default=list)
    success = ArrayField(models.BooleanField(), default=list)
    n_seen = ArrayField(models.IntegerField(), default=list)

    seed = models.IntegerField()

    @property
    def post(self):

        post_entries = self.post_set.all()

        post = {stat: {} for stat in ("std", "mean")}

        for pr in self.param_labels:
            post_entry_pr = post_entries.get(param_label=pr)
            post["mean"][pr] = np.array(post_entry_pr.mean[:self.n_session])
            post["std"][pr] = np.array(post_entry_pr.std[:self.n_session])

        return post

    @property
    def timestamp_array(self):
        return np.asarray(self.timestamp[:self.n_session], dtype=int)

    @property
    def hist_array(self):
        return np.asarray(self.hist[:self.n_session], dtype=int)

    @property
    def success_array(self):
        return np.asarray(self.success[:self.n_session], dtype=bool)

    @property
    def n_seen_array(self):
        return np.asarray(self.n_seen[:self.n_session], dtype=int)


class Post(models.Model):

    simulation = models.ForeignKey(Simulation, on_delete=models.CASCADE)
    param_label = models.CharField(max_length=32)
    mean = ArrayField(models.FloatField(), default=list)
    std = ArrayField(models.FloatField(), default=list)


class RandomState(models.Model):

    simulation = models.OneToOneField(Simulation, on_delete=models.CASCADE)
    entry_1 = models.TextField()
    entry_2 = ArrayField(models.IntegerField())
    entry_3 = models.IntegerField()
    entry_4 = models.IntegerField()
    entry_5 = models.FloatField()


# class ProbRecall(models.Model):
#
#     simulation = models.ForeignKey(Simulation, on_delete=models.CASCADE)
#     item = models.IntegerField()
#     p = models.FloatField()
#     iteration = models.IntegerField()
