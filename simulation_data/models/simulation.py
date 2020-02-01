from django.db import models

from django.contrib.postgres.fields import ArrayField


# Create your models here.
class Simulation(models.Model):

    param = ArrayField(models.IntegerField(), default=list)