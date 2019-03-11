from gensim.models import KeyedVectors

import pickle
import os

from itertools import combinations
from datetime import datetime

import uuid


def _load_model(model_path='data/word_vectors.kv'):

    t0 = datetime.utcnow()

    if not os.path.exists(model_path):
        model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
        model.save(model_path)

    else:
        model = KeyedVectors.load(model_path, mmap='r')

    t1 = datetime.utcnow()

    print(f"Time to load: {t1 - t0}")

    return model


def get_semantic_distance(word_list):

    word_list = word_list.sort()

    list_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, f"{word_list}"))

    backup_dic = f"data/{list_id}.p"

    if not os.path.exists(backup_dic):

        model = _load_model()

        dic = {}

        for a, b in combinations(word_list, 2):

            distance = model.distance(a, b)
            dic[(a, b)] = distance
            dic[(b, a)] = distance
            print(f"Distance between {a} & {b}: {distance}")

        pickle.dump(dic, file=open(backup_dic, 'wb'))
