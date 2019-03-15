from gensim.models import KeyedVectors

import os
import numpy as np

from itertools import combinations
from datetime import datetime

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
BACKUP_FOLDER = f'{SCRIPT_FOLDER}/backup'
DATA_FOLDER = f'{SCRIPT_FOLDER}/data'

DATA = f'{DATA_FOLDER}/GoogleNews-vectors-negative300.bin'
MODEL = f'{BACKUP_FOLDER}/word_vectors.kv'


def _load_model():

    t0 = datetime.utcnow()

    print('Load Word2Vec model...', end=" ")
    if not os.path.exists(MODEL):
        model = KeyedVectors.load_word2vec_format(DATA, binary=True)
        model.save(MODEL)

    else:
        model = KeyedVectors.load(MODEL, mmap='r')

    t1 = datetime.utcnow()

    print(f"Done in {t1 - t0}")

    return model


def evaluate_similarity(word_list):

    model = _load_model()

    n_word = len(word_list)

    sim = np.zeros((n_word, n_word))

    for a, b in combinations(word_list, 2):

        similarity = model.similarity(a, b)

        i = word_list.index(a)
        j = word_list.index(b)

        sim[i, j] = similarity
        sim[j, i] = similarity

        # print(f"Distance between {a} & {b}: {distance}")

    return similarity
