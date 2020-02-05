from django.db import models

from django.contrib.postgres.fields import ArrayField


# Create your models here.
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


class Post(models.Model):

    simulation = models.ForeignKey(Simulation, on_delete=models.CASCADE)
    param_label = models.CharField(max_length=32)
    mean = ArrayField(models.FloatField(), default=list)
    std = ArrayField(models.FloatField(), default=list)


# class ProbRecall(models.Model):
#
#     simulation = models.ForeignKey(Simulation, on_delete=models.CASCADE)
#     item = models.IntegerField()
#     p = models.FloatField()
#     iteration = models.IntegerField()
