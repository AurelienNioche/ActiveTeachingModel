import os
import pickle
# import uuid
from itertools import combinations

import numpy as np

from user_data.settings import BKP_CONNECTION


def create(word_list, use_nan):

    from . word2vec import word2vec
    sim = word2vec.evaluate_similarity(word_list=word_list, use_nan=use_nan)
    return sim


def _normalize(a):
    return np.interp(a, (np.nanmin(a), np.nanmax(a)), (0, 1))


def _compute(word_list, normalize_similarity, verbose=False):

    word_list = [i.lower() for i in word_list]

    sim = create(word_list=word_list, use_nan=normalize_similarity)
    if normalize_similarity:
        sim = _normalize(sim)

    if verbose:
        for i, j in combinations(range(len(word_list)), r=2):
            if i != j:
                a = word_list[i]
                b = word_list[j]
                similarity = sim[i, j]
                print(f"Similarity between {a} and {b} is: {similarity:.2f}")

    return sim


def get(word_list, normalize_similarity):

    bkp_file = os.path.join(BKP_CONNECTION,
                            f"sem_con_norm_{normalize_similarity}.p")

    if os.path.exists(bkp_file):

        data, loaded_word_list, loaded_normalize = \
            pickle.load(open(bkp_file, 'rb'))

        conform = (loaded_word_list == np.array(word_list)).all() \
            and loaded_normalize == normalize_similarity

        if conform:
            return data

    assert word_list is not None, \
        "If no backup file is existing, a word list has to be provided "

    data = _compute(word_list=word_list,
                    normalize_similarity=normalize_similarity)
    os.makedirs(BKP_CONNECTION, exist_ok=True)
    pickle.dump((data, word_list, normalize_similarity),
                open(bkp_file, 'wb'))
    return data


def demo(word_list=None):

    if not word_list:
        word_list = ['computer', 'myself', 'king']

    _compute(word_list=word_list, normalize_similarity=True,
             verbose=True)
