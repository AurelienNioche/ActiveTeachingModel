import os

# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

# Your application specific imports
from task.models import Question, User

from utils.django import AskUser


@AskUser
def _load_user_data():
    Question.objects.all().delete()
    User.objects.all().delete()
    os.system('psql ActiveTeaching < data/user_and_question_tables.sql')


def load_user_data():

    print("Warning: loading user data will erase previous data if any.")
    _load_user_data()


if __name__ == "__main__":
    load_user_data()
