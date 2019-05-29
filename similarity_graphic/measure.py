# import os
#
# # Django specific settings
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# # Ensure settings are read
# from django.core.wsgi import get_wsgi_application
#
# application = get_wsgi_application()
#
# # Your application specific imports
# from task.models import Kanji

import numpy as np
# import pickle

from itertools import combinations
# import uuid


from similarity_graphic.auto_encoder import auto_encoder
from similarity_graphic.sim_search import sim_search


# SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
# BACKUP_FOLDER = f'{SCRIPT_FOLDER}/backup'
# GRAPHIC_PROPERTIES = f'{BACKUP_FOLDER}/graphic_properties.p'


# def _normalize(a):
#     return np.interp(a, (np.nanmin(a), np.nanmax(a)), (0, 1))
    # return # (x - np.min(x)) / (np.max(x) - np.min(x))


def get(kanji_list, method='sim_search', verbose=False):

    if method == 'auto_encoder':
        s = auto_encoder.get_similarity()
    elif method == 'sim_search':
        s = sim_search.get_similarity()
    else:
        raise Exception(f'Method {method} is not implemented')

    n_kanji = len(kanji_list)
    sim_array = np.zeros((n_kanji, n_kanji))
    for i, j in combinations(range(n_kanji), r=2):

        a = kanji_list[i]
        b = kanji_list[j]
        similarity = s(a, b)

        sim_array[i, j] = sim_array[j, i] = similarity
        if verbose:
            print(f"Similarity between {a} and {b} is: {similarity:.2f}")

    return sim_array

    # # list_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, f"{kanji_list}"))
    # # backup = f"{BACKUP_FOLDER}/{list_id}.p"
    #
    # if not os.path.exists(backup) or force:
    #     sim = create(kanji_list=kanji_list, force=force)
    #     pickle.dump(obj=sim, file=open(backup, 'wb'))
    #
    # else:
    #     sim = pickle.load(file=open(backup, 'rb'))
    #
    # if normalize:
    #     sim = _normalize(sim)
    #
    # if verbose:
    #     for i, j in combinations(range(len(kanji_list)), r=2):
    #         a = kanji_list[i]
    #         b = kanji_list[j]
    #         similarity = sim[i, j]
    #         print(f"Similarity between {a} and {b} is: {similarity:.2f}")
    #
    # return sim


# def get_distance(kanji1, kanji2):
#
#     graphic_prop = get_graphic_properties()
#
#     x = graphic_prop[kanji1]
#     y = graphic_prop[kanji2]
#
#     # print(x.shape)
#
#     distance = np.abs(np.linalg.norm(x - y))
#     print(f"Distance between {kanji1} and {kanji2} is {distance:.3f}")


def demo(method='auto_encoder'):

    if method == 'auto_encoder':
        s = auto_encoder.get_similarity()
    elif method == 'sim_search':
        s = sim_search.get_similarity()
    else:
        raise Exception(f'Method {method} is not implemented')

    # from auto_encoder import auto_encoder
    #
    # auto_encoder.evaluate_model()

    # create_graphic_properties()
    for a, b in combinations(['目', '耳', '犬', '大', '王', '玉', '夕', '二', '一',
                              '左', '右'], r=2):
        similarity = s(a, b)
        print(f"Similarity between {a} and {b} is: {similarity:.2f}")


if __name__ == "__main__":

    demo()

    # create_graphic_properties(force=True)
