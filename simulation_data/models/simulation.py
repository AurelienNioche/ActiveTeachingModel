from django.db import models

from django.contrib.postgres.fields import ArrayField


# Create your models here.
class Simulation(models.Model):

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


class Post(models.Model):

    simulation = models.ForeignKey(Simulation, on_delete=models.CASCADE)
    param_label = models.CharField(max_length=32)
    iteration = models.IntegerField()
    mean = models.FloatField()
    std = models.FloatField()


class Iteration(models.Model):

    simulation = models.ForeignKey(Simulation, on_delete=models.CASCADE)
    item = models.IntegerField()
    iteration = models.IntegerField()
    timestamp = models.IntegerField()
    success = models.BooleanField()
    n_seen = models.IntegerField()


class ProbRecall(models.Model):

    simulation = models.ForeignKey(Simulation, on_delete=models.CASCADE)
    item = models.IntegerField()
    p = models.FloatField()
    iteration = models.IntegerField()