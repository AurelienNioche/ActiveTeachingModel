import os
import pickle

import numpy as np

from itertools import combinations
from tqdm import tqdm

from . simsearch import simsearch

from user_data.settings import BKP_CONNECTION


def _normalize(a):
    return np.interp(a, (np.nanmin(a), np.nanmax(a)), (0, 1))
    # return # (x - np.min(x)) / (np.max(x) - np.min(x))


def _compute(kanji_list, normalize_similarity, verbose=False):

    s = simsearch.StrokeSimilarity()

    n_kanji = len(kanji_list)
    sim_array = np.zeros((n_kanji, n_kanji))

    comb = list(combinations(range(n_kanji), r=2))
    for i, j in tqdm(comb):

        if i == j and normalize_similarity:
            similarity = np.nan
        else:
            a = kanji_list[i]
            b = kanji_list[j]
            similarity = s(a, b)

        sim_array[i, j] = sim_array[j, i] = similarity

    if normalize_similarity:
        sim_array = _normalize(sim_array)

    if verbose:
        for i, j in combinations(range(n_kanji), r=2):
            if i != j:
                similarity = sim_array[i, j]
                a = kanji_list[i]
                b = kanji_list[j]
                print(f"Similarity between {a} and {b} is: {similarity:.2f}")

    return sim_array


def get(kanji_list, normalize_similarity):

    bkp_file = os.path.join(BKP_CONNECTION,
                            f"graph_con_norm_{normalize_similarity}.p")

    if os.path.exists(bkp_file):

        data, loaded_kanji_list, loaded_normalize = \
            pickle.load(open(bkp_file, 'rb'))

        conform = (loaded_kanji_list == np.array(kanji_list)).all() \
            and loaded_normalize == normalize_similarity

        if conform:
            return data

    assert kanji_list is not None, \
        "If no backup file is existing, a kanji list has to be provided "

    data = _compute(kanji_list=kanji_list,
                    normalize_similarity=normalize_similarity)
    os.makedirs(BKP_CONNECTION, exist_ok=True)
    pickle.dump((data, kanji_list, normalize_similarity),
                open(bkp_file, 'wb'))

    return data


def demo(kanji_list=None):

    if not kanji_list:
        kanji_list = ['目', '耳', '犬', '大', '王',
                      '玉', '夕', '二', '一', '左', '右']

    _compute(kanji_list=kanji_list, normalize_similarity=True,
             verbose=True)
