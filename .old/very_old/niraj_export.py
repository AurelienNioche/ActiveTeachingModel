import os
# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

# Your application specific imports
from task.models import User

import pickle

import behavior.aalto

import similarity_graphic.measure
import similarity_semantic.measure


DATA_FOLDER = "niraj_export"


os.makedirs(DATA_FOLDER, exist_ok=True)


def export():

    users = User.objects.all().order_by('id')

    for u in users:

        print(f'{u.id}\n{"*" * 5}\n')

        # Get questions, replies, possible_replies, and number of different items
        questions, replies, n_items, possible_replies, success = behavior.aalto.get(user_id=u.id, verbose=True)

        # Get task parameters for ACT-R +
        question_entries, kanjis, meanings = behavior.aalto.task_features(user_id=u.id)

        c_graphic = similarity_graphic.measure.get(kanjis)
        c_semantic = similarity_semantic.measure.get(meanings)

        pickle.dump(questions, open(f'{DATA_FOLDER}/q_user{u.id}.p', 'wb'))
        pickle.dump(success, open(f'{DATA_FOLDER}/r_user{u.id}.p', 'wb'))
        pickle.dump(c_graphic, open(f'{DATA_FOLDER}/g_user{u.id}.p', 'wb'))
        pickle.dump(c_semantic, open(f'{DATA_FOLDER}/s_user{u.id}.p', 'wb'))


if __name__ == "__main__":

    export()
