from django.db import models


# Create your models here.
class Item(models.Model):

    p_recall = models.FloatField(default=-1)
    timestamp = models.IntegerField(default=-1)
    delta = models.IntegerField(default=-1)
    user_id = models.CharField(max_length=255, blank=True, null=True,
                               default='<empty>')
    learning_language = models.CharField(max_length=2, blank=True, null=True,
                                         default='<empty>')
    ui_language = models.CharField(max_length=2, blank=True, null=True,
                                   default='<empty>')
    lexeme_id = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    lexeme_string = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    history_seen = models.IntegerField(default=-1)
    history_correct = models.IntegerField(default=-1)
    session_seen = models.IntegerField(default=-1)
    session_correct = models.IntegerField(default=-1)

    class Meta:

        db_table = 'item'
        app_label = 'duolingo'
