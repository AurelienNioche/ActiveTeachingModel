import os
import django.db.utils
import psycopg2

# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingServer.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

# Your application specific imports
from task.models import Kanji, Parameter

from utils import AskUser


@AskUser
def fill_kanji_table():

    Kanji.objects.all().delete()
    os.system('psql ActiveTeaching < data/kanji_table.sql')


def main():

    print("I will fill the kanji table and add default parameters.")
    print('Warning: Filling the kanji table will erase actual content if any.')

    fill_kanji_table()


if __name__ == "__main__":

    main()
