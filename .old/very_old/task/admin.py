from django.contrib import admin

# Register your models here.
from . models import User, Question, Kanji, Parameter, PredefinedQuestion

from . parameters import N_POSSIBLE_REPLIES


class UserAdmin(admin.ModelAdmin):
    list_display = (
        "id", "registration_time")  # "username", "gender", "age", "mother_tongue", "other_language",


class QuestionAdmin(admin.ModelAdmin):
    list_display = (
        "user_id", "t", "question", "correct_answer", "reply", "time_display", "time_reply") \
        + tuple(f"possible_reply_{i}" for i in range(N_POSSIBLE_REPLIES))


class KanjiAdmin(admin.ModelAdmin):
    list_display = (
        "id", "kanji", "meaning", "translation_of_on", "translation_of_kun", "grade")
    # , "reading_within_joyo", "reading_beyond_joyo")


class ParameterAdmin(admin.ModelAdmin):
    list_display = (
        "name", "value")


class PredefinedQuestionAdmin(admin.ModelAdmin):
    list_display = (
        "t", "question", "correct_answer") \
        + tuple(f"possible_reply_{i}" for i in range(N_POSSIBLE_REPLIES))


admin.site.register(User, UserAdmin)
admin.site.register(Question, QuestionAdmin)
admin.site.register(Kanji, KanjiAdmin)
admin.site.register(Parameter, ParameterAdmin)
admin.site.register(PredefinedQuestion, PredefinedQuestionAdmin)
