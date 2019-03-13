import os

# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

# Your application specific imports
from task.models import User, Question, Kanji

import pickle
import numpy as np
from itertools import combinations

from analysis import get_task_features


BACKUP_FOLDER = 'backup'


def create_dic_sim(dic_sim_file):

    from auto_encoder import auto_encoder

    kanji_entries = [k for k in Kanji.objects.all().order_by('id')]

    kanji_dic = {k.id: k.kanji for k in kanji_entries}

    autoencoder, encoder, decoder = auto_encoder.get(kanji_dic=kanji_dic)

    os.makedirs(BACKUP_FOLDER, exist_ok=True)

    n_kanji = len(kanji_entries)

    dim_entry = encoder.layers[0].input_shape[1:]
    # dim_encoding = encoder.layers[-1].output_shape[1:]

    # print(dim_encoding)

    a = np.zeros((n_kanji,) + dim_entry)

    for i, k in enumerate(kanji_entries):
        a[i] = auto_encoder.get_formatted_image_for_cnn(k.id)

    v = encoder.predict(a)

    sim_dic = {}

    for i, k in enumerate(kanji_entries):
        encoded = v[i]
        sim_dic[k.kanji] = encoded

    pickle.dump(obj=sim_dic, file=open(dic_sim_file, 'wb'))

    return sim_dic


def main():

    dic_sim_file = f'{BACKUP_FOLDER}/dic_sim.p'

    if not os.path.exists(dic_sim_file):
        sim_dic = create_dic_sim(dic_sim_file)

    else:
        sim_dic = pickle.load(file=open(dic_sim_file, 'rb'))

    users = User.objects.all().order_by('id')

    bic_data = [[], []]

    for u in users:

        question_entries, kanjis, meanings = get_task_features(user_id=u.id)

        # n_kanji = len(kanjis)
        # a = np.zeros((n_kanji, n_kanji))

        dic = {}

        for a, b in combinations(kanjis, 2):

            x = sim_dic[a]
            y = sim_dic[b]

            distance = np.abs(np.linalg.norm(x - y))

            dic[(a, b)] = distance
            dic[(b, a)] = distance
            print(f"Distance between {a} & {b}: {distance}")


if __name__ == "__main__":

    main()
