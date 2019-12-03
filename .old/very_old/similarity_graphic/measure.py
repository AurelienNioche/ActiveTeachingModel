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


def _normalize(a):
    return np.interp(a, (np.nanmin(a), np.nanmax(a)), (0, 1))
    # return # (x - np.min(x)) / (np.max(x) - np.min(x))


def get(kanji_list, method='sim_search',  normalize_similarity=False, verbose=False):

    if method == 'auto_encoder':
        from similarity_graphic.auto_encoder import auto_encoder
        s = auto_encoder.get_similarity()
    elif method == 'sim_search':
        from similarity_graphic.sim_search import sim_search
        s = sim_search.get_similarity()
    else:
        raise Exception(f'Method {method} is not implemented')

    n_kanji = len(kanji_list)
    sim_array = np.zeros((n_kanji, n_kanji))
    for i, j in combinations(range(n_kanji), r=2):

        if i == j and normalize_similarity:
            similarity = np.nan
        else:
            a = kanji_list[i]
            b = kanji_list[j]
            similarity = s(a, b)

        sim_array[i, j] = sim_array[j, i] = similarity
        if verbose and i != j:
            print(f"Similarity between {a} and {b} is: {similarity:.2f}")

    if normalize_similarity:
        sim_array = _normalize(sim_array)

    return sim_array


def demo(method='auto_encoder'):

    if method == 'auto_encoder':
        from similarity_graphic.auto_encoder import auto_encoder
        s = auto_encoder.get_similarity()
    elif method == 'sim_search':
        from similarity_graphic.sim_search import sim_search
        s = sim_search.get_similarity()
    else:
        raise Exception(f'Method {method} is not implemented')

    for a, b in combinations(['目', '耳', '犬', '大', '王', '玉', '夕', '二', '一',
                              '左', '右'], r=2):
        similarity = s(a, b)
        print(f"Similarity between {a} and {b} is: {similarity:.2f}")


if __name__ == "__main__":

    demo()
