from django.utils.timezone import now

from django.db import models

# Create your models here.


class Question(models.Model):

    user_id = models.IntegerField(default=-1)
    t = models.IntegerField(default=-1)
    question = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    correct_answer = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    possible_reply_0 = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    possible_reply_1 = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    possible_reply_2 = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    possible_reply_3 = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    possible_reply_4 = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    possible_reply_5 = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    reply = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    time_display = models.DateTimeField(default=now)
    time_reply = models.DateTimeField(default=now)

    class Meta:

        db_table = 'question'
        app_label = 'task'


class User(models.Model):

    # username = models.TextField(unique=True)
    # gender = models.TextField(blank=True, null=True)
    # age = models.IntegerField(blank=True, null=True)
    # mother_tongue = models.TextField(blank=True, null=True)
    # other_language = models.TextField(blank=True, null=True)
    # t = models.IntegerField(blank=True, null=True, default=0)
    registration_time = models.DateTimeField(default=now)

    class Meta:
        db_table = 'user'
        app_label = 'task'


class Parameter(models.Model):

    name = models.CharField(max_length=255, null=False, unique=True)
    value = models.IntegerField(default=-1)

    class Meta:
        db_table = 'parameter'
        app_label = 'task'


class PredefinedQuestion(models.Model):

    t = models.IntegerField(default=-1)
    question = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    correct_answer = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    possible_reply_0 = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    possible_reply_1 = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    possible_reply_2 = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    possible_reply_3 = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    possible_reply_4 = models.CharField(max_length=255, blank=True, null=True, default='<empty>')
    possible_reply_5 = models.CharField(max_length=255, blank=True, null=True, default='<empty>')

    class Meta:
        db_table = 'predefined_question'
        app_label = 'task'


class Kanji(models.Model):

    id = models.IntegerField(primary_key=True)
    kanji = models.CharField(db_column='Kanji', max_length=255, blank=True, null=True)  # Field name made lowercase.
    meaning = models.CharField(max_length=2555, blank=True, null=True)

    strokes = models.IntegerField(db_column='Strokes', blank=True, null=True)  # Field name made lowercase.
    grade = models.IntegerField(db_column='Grade', blank=True, null=True)  # Field name made lowercase.
    kanji_classification = models.CharField(db_column='Kanji Classification', max_length=255, blank=True,
                                            null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    jlpt_test = models.CharField(db_column='JLPT-test', max_length=255, blank=True,
                                 null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    name_of_radical = models.CharField(db_column='Name of Radical', max_length=255, blank=True,
                                       null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    radical_freq_field = models.IntegerField(db_column='Radical Freq.', blank=True,
                                             null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    reading_within_joyo = models.CharField(db_column='Reading within Joyo', max_length=255, blank=True,
                                           null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    reading_beyond_joyo = models.CharField(db_column='Reading beyond Joyo', max_length=255, blank=True,
                                           null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    field_of_on = models.IntegerField(db_column='# of On', blank=True,
                                      null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it started with '_'.
    on_within_joyo = models.CharField(db_column='On within Joyo', max_length=255, blank=True,
                                      null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    kanji_id_in_nelson = models.IntegerField(db_column='Kanji ID in Nelson', blank=True,
                                             null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    field_of_meanings_of_on = models.IntegerField(db_column='# of Meanings of On', blank=True,
                                                  null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it started with '_'.
    translation_of_on = models.TextField(db_column='Translation of On', blank=True,
                                         null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    field_of_kun_within_joyo_with_inflections = models.IntegerField(db_column='# of Kun within Joyo with inflections',
                                                                    blank=True,
                                                                    null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it started with '_'.
    field_of_kun_within_joyo_without_inflections = models.IntegerField(
        db_column='# of Kun within Joyo without inflections', blank=True,
        null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it started with '_'.
    kun_within_joyo = models.CharField(db_column='Kun within Joyo', max_length=255, blank=True,
                                       null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    field_of_meanings_of_kun = models.IntegerField(db_column='# of Meanings of Kun', blank=True,
                                                   null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it started with '_'.
    translation_of_kun = models.TextField(db_column='Translation of Kun', blank=True,
                                          null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    year_of_inclusion = models.IntegerField(db_column='Year of Inclusion', blank=True,
                                            null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    kanji_frequency_with_proper_nouns = models.IntegerField(db_column='Kanji Frequency with Proper Nouns', blank=True,
                                                            null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    acc_freq_on_with_proper_nouns = models.IntegerField(db_column='Acc. Freq. On with Proper Nouns', blank=True,
                                                        null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    acc_freq_kun_with_proper_nouns = models.IntegerField(db_column='Acc. Freq. Kun with Proper Nouns', blank=True,
                                                         null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    on_ratio_with_proper_nouns = models.DecimalField(db_column='On Ratio with Proper Nouns', max_digits=4,
                                                     decimal_places=3, blank=True,
                                                     null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    acc_freq_on_beyond_joyo_with_proper_nouns = models.IntegerField(
        db_column='Acc. Freq. On beyond Joyo with Proper Nouns', blank=True,
        null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    acc_freq_kun_beyond_joyo_with_proper_nouns = models.IntegerField(
        db_column='Acc. Freq. Kun beyond Joyo with Proper Nouns', blank=True,
        null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    acc_on_ratio_beyond_joyo_with_proper_nouns = models.DecimalField(
        db_column='Acc. On Ratio beyond Joyo with Proper Nouns', max_digits=4, decimal_places=3, blank=True,
        null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    kanji_frequency_without_proper_nouns = models.IntegerField(db_column='Kanji Frequency without Proper Nouns',
                                                               blank=True,
                                                               null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    acc_freq_on_without_proper_nouns = models.IntegerField(db_column='Acc. Freq. On without Proper Nouns', blank=True,
                                                           null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    acc_freq_kun_without_proper_nouns = models.IntegerField(db_column='Acc. Freq. Kun without Proper Nouns', blank=True,
                                                            null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    on_ratio_without_proper_nouns = models.DecimalField(db_column='On Ratio without Proper Nouns', max_digits=1000,
                                                        decimal_places=999, blank=True,
                                                        null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    acc_freq_on_beyond_joyo_without_proper_nouns = models.IntegerField(
        db_column='Acc. Freq. On beyond Joyo without Proper Nouns', blank=True,
        null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    acc_freq_kun_beyond_joyo_without_proper_nouns = models.IntegerField(
        db_column='Acc. Freq. Kun beyond Joyo without Proper Nouns', blank=True,
        null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    on_ratio_beyond_joyo_without_proper_nouns = models.DecimalField(
        db_column='On Ratio beyond Joyo without Proper Nouns', max_digits=4, decimal_places=3, blank=True,
        null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    left_kanji_prod_field = models.IntegerField(db_column='Left Kanji Prod.', blank=True,
                                                null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    right_kanji_prod_field = models.IntegerField(db_column='Right Kanji Prod.', blank=True,
                                                 null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    acc_freq_left_prod_field = models.IntegerField(db_column='Acc. Freq. Left Prod.', blank=True,
                                                   null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    acc_freq_right_prod_field = models.IntegerField(db_column='Acc. Freq. Right Prod.', blank=True,
                                                    null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters. Field renamed because it ended with '_'.
    symmetry = models.CharField(db_column='Symmetry', max_length=255, blank=True,
                                null=True)  # Field name made lowercase.
    left_entropy = models.DecimalField(db_column='Left Entropy', max_digits=10, decimal_places=9, blank=True,
                                       null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    right_entropy = models.DecimalField(db_column='Right Entropy', max_digits=10, decimal_places=9, blank=True,
                                        null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    left1sound = models.CharField(db_column='Left1sound', max_length=255, blank=True,
                                  null=True)  # Field name made lowercase.
    left1freq = models.IntegerField(db_column='Left1freq', blank=True, null=True)  # Field name made lowercase.
    left2sound = models.CharField(db_column='Left2sound', max_length=255, blank=True,
                                  null=True)  # Field name made lowercase.
    left2freq = models.IntegerField(db_column='Left2freq', blank=True, null=True)  # Field name made lowercase.
    left3sound = models.CharField(db_column='Left3sound', max_length=255, blank=True,
                                  null=True)  # Field name made lowercase.
    left3freq = models.IntegerField(db_column='Left3freq', blank=True, null=True)  # Field name made lowercase.
    left4sound = models.CharField(db_column='Left4sound', max_length=255, blank=True,
                                  null=True)  # Field name made lowercase.
    left4freq = models.IntegerField(db_column='Left4freq', blank=True, null=True)  # Field name made lowercase.
    left5sound = models.CharField(db_column='Left5sound', max_length=255, blank=True,
                                  null=True)  # Field name made lowercase.
    left5freq = models.IntegerField(db_column='Left5freq', blank=True, null=True)  # Field name made lowercase.
    left6sound = models.CharField(db_column='Left6sound', max_length=255, blank=True,
                                  null=True)  # Field name made lowercase.
    left6freq = models.IntegerField(db_column='Left6freq', blank=True, null=True)  # Field name made lowercase.
    right1sound = models.CharField(db_column='Right1sound', max_length=255, blank=True,
                                   null=True)  # Field name made lowercase.
    right1freq = models.IntegerField(db_column='Right1freq', blank=True, null=True)  # Field name made lowercase.
    right2sound = models.CharField(db_column='Right2sound', max_length=255, blank=True,
                                   null=True)  # Field name made lowercase.
    right2freq = models.IntegerField(db_column='Right2freq', blank=True, null=True)  # Field name made lowercase.
    right3sound = models.CharField(db_column='Right3sound', max_length=255, blank=True,
                                   null=True)  # Field name made lowercase.
    right3freq = models.IntegerField(db_column='Right3freq', blank=True, null=True)  # Field name made lowercase.
    right4sound = models.CharField(db_column='Right4sound', max_length=255, blank=True,
                                   null=True)  # Field name made lowercase.
    right4freq = models.IntegerField(db_column='Right4freq', blank=True, null=True)  # Field name made lowercase.
    right5sound = models.CharField(db_column='Right5sound', max_length=255, blank=True,
                                   null=True)  # Field name made lowercase.
    right5freq = models.IntegerField(db_column='Right5freq', blank=True, null=True)  # Field name made lowercase.
    right6sound = models.CharField(db_column='Right6sound', max_length=255, blank=True,
                                   null=True)  # Field name made lowercase.
    right6freq = models.IntegerField(db_column='Right6freq', blank=True, null=True)  # Field name made lowercase.
    right7sound = models.CharField(db_column='Right7sound', max_length=255, blank=True,
                                   null=True)  # Field name made lowercase.
    right7freq = models.IntegerField(db_column='Right7freq', blank=True, null=True)  # Field name made lowercase.

    class Meta:

        db_table = 'kanji'
        app_label = 'task'
