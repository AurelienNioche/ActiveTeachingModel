import os

# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

# Your application specific imports
from task.models import User, Question, Kanji

from auto_encoder.create_images import create_image_from_character


def main():

    kanji = [(k.id, k.kanji) for k in Kanji.objects.all().order_by('id')]
    for (i, k) in kanji:

        create_image_from_character(k, name=i, size=100)


if __name__ == "__main__":

    main()
