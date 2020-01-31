import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "ActiveTeachingServer.settings")
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

from ActiveTeachingModel.settings import DATABASES
from django.contrib.auth.models import User

DB_NAME = DATABASES['default']['NAME']


# entries = Kanji.objects.all().order_by('id')
# for e in entries:
#
#     m_entries = Meaning.objects.filter(meaning=e.meaning_string)
#     if m_entries:
#         m = m_entries[0]
#     else:
#         m = Meaning.objects.create(meaning=e.meaning_string)
#
#     e.meaning = m
#     e.save()


def main():
    os.system(f"createdb {DB_NAME} --owner postgres")
    os.system("python3 manage.py makemigrations")
    os.system("python3 manage.py migrate")
    User.objects.create_superuser(f'{HOST_USER}',
                                  f'{HOST_PASSWORD}')


if __name__ == "__main__":
    main()
