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


def create_graphic_properties(force=False):

    from auto_encoder import auto_encoder

    kanji_entries = Kanji.objects.all().order_by('id')
    kanji_dic = {k.id: k.kanji for k in kanji_entries}

    # Get the trained models
    (autoencoder, encoder, decoder), history = \
        auto_encoder.get_models(kanji_dic=kanji_dic, force=force)

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


def create(kanji_list, force):

    graphic_prop = get_graphic_properties()

    n_kanji = len(kanji_list)

    dist = np.zeros((n_kanji, n_kanji))

    for i in range(n_kanji):
        dist[i, i] = np.nan

    for a, b in combinations(kanji_list, 2):

        x = graphic_prop[a]
        y = graphic_prop[b]

        # print(x.shape)

        distance = np.abs(np.linalg.norm(x - y))

        i = kanji_list.index(a)
        j = kanji_list.index(b)

        dist[i, j] = distance
        dist[j, i] = distance

    dist /= np.nanmax(dist)

    sim = 1 - dist

    return sim


def _normalize(a):
    return np.interp(a, (np.nanmin(a), np.nanmax(a)), (0, 1))
    # return # (x - np.min(x)) / (np.max(x) - np.min(x))


def get(kanji_list, normalize=True, force=False, verbose=False):

    list_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, f"{kanji_list}"))
    backup = f"{BACKUP_FOLDER}/{list_id}.p"

    if not os.path.exists(backup) or force:
        sim = create(kanji_list=kanji_list, force=force)
        pickle.dump(obj=sim, file=open(backup, 'wb'))

    else:
        sim = pickle.load(file=open(backup, 'rb'))

    if normalize:
        sim = _normalize(sim)

    if verbose:
        for i, j in combinations(range(len(kanji_list)), r=2):
            a = kanji_list[i]
            b = kanji_list[j]
            similarity = sim[i, j]
            print(f"Similarity between {a} and {b} is: {similarity:.2f}")

    return sim


def get_distance(kanji1, kanji2):

    graphic_prop = get_graphic_properties()

    x = graphic_prop[kanji1]
    y = graphic_prop[kanji2]

    # print(x.shape)

    distance = np.abs(np.linalg.norm(x - y))
    print(f"Distance between {kanji1} and {kanji2} is {distance:.3f}")


def main():

    # from auto_encoder import auto_encoder
    #
    # auto_encoder.evaluate_model()

    # create_graphic_properties()
    for a, b in combinations(['目', '耳', '犬', '大', '王', '玉', '夕', '二', '一',
                              '左', '右'], r=2):
        get_distance(a, b)


if __name__ == "__main__":

    main()

    # create_graphic_properties(force=True)
