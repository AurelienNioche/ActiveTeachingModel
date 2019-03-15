import os

# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

# Your application specific imports
from task.models import Kanji

import numpy as np
import pickle

from itertools import combinations
import uuid


SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
BACKUP_FOLDER = f'{SCRIPT_FOLDER}/backup'
GRAPHIC_PROPERTIES = f'{BACKUP_FOLDER}/graphic_properties.p'


def create_graphic_properties():

    # Aviut ti ulirt
    from auto_encoder import auto_encoder

    kanji_entries = Kanji.objects.all().order_by('id')
    kanji_dic = {k.id: k.kanji for k in kanji_entries}

    # Get the trained models
    autoencoder, encoder, decoder = auto_encoder.get_models(kanji_dic=kanji_dic)

    n_kanji = len(kanji_dic)

    # Dimension of the entry of the encoder
    dim_entry = encoder.layers[0].input_shape[1:]

    # Get the images formatted for the autoencoder
    a = np.zeros((n_kanji,) + dim_entry)

    keys = kanji_dic.keys()

    for i, k in enumerate(keys):
        a[i] = auto_encoder.get_formatted_image_for_cnn(k)

    # Get the encoded value
    v = encoder.predict(a)

    # Create a dictionary with graphic representation of all kanji
    graphic_prop = {}

    for i, k in enumerate(keys):
        encoded = v[i]
        graphic_prop[kanji_dic[k]] = encoded

    # Save in pickle
    pickle.dump(obj=graphic_prop, file=open(GRAPHIC_PROPERTIES, 'wb'))

    return graphic_prop


def get_graphic_properties():

    os.makedirs(BACKUP_FOLDER, exist_ok=True)

    if not os.path.exists(GRAPHIC_PROPERTIES):
        graphic_prop = create_graphic_properties()

    else:
        graphic_prop = pickle.load(file=open(GRAPHIC_PROPERTIES, 'rb'))

    return graphic_prop


def create(kanji_list, backup):

    graphic_prop = get_graphic_properties()

    n_kanji = len(kanji_list)

    dist = np.zeros((n_kanji, n_kanji))

    for a, b in combinations(kanji_list, 2):

        x = graphic_prop[a]
        y = graphic_prop[b]

        # print(x.shape)

        distance = np.abs(np.linalg.norm(x - y))

        i = kanji_list.index(a)
        j = kanji_list.index(b)

        dist[i, j] = distance
        dist[j, i] = distance

    #     print(f"Distance between {a} & {b}: {distance}")
    #
    # print()
    # for a, b in combinations(kanji_list, 2):
    #     x = graphic_prop[a]
    #     y = graphic_prop[b]
    #
    #     distance = np.abs(np.linalg.norm(x) - np.linalg.norm(y))
    #     print(f"Distance between {a} & {b}: {distance}")
    #
    # print()
    # for a, b in combinations(kanji_list, 2):
    #     x = graphic_prop[a]
    #     y = graphic_prop[b]
    #
    #     distance = 0
    #
    #     for i in range(x.shape[0]):
    #         for j in range(x.shape[1]):
    #             for k in range(x.shape[2]):
    #                 distance += (x[i, j, k] - y[i, j, k])**2
    #
    #     distance = distance**(1/2)
    #
    #     print(f"Distance between {a} & {b}: {distance}")

    dist /= np.max(dist)

    sim = 1 - dist

    # Save in pickle
    pickle.dump(obj=sim, file=open(backup, 'wb'))

    return sim


def get(kanji_list, force=False):

    list_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, f"{kanji_list}"))
    backup = f"{BACKUP_FOLDER}/{list_id}.p"

    if not os.path.exists(backup) or force:
        sim = create(kanji_list=kanji_list, backup=backup)

    else:
        sim = pickle.load(file=open(backup, 'rb'))

    return sim
