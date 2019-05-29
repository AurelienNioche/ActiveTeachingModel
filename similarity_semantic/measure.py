import numpy as np
import os
import pickle
import uuid
from itertools import combinations

from similarity_semantic.word2vec import word2vec

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
BACKUP_FOLDER = f'{SCRIPT_FOLDER}/backup'

os.makedirs(BACKUP_FOLDER, exist_ok=True)


def create(word_list, backup, use_nan):

    sim = word2vec.evaluate_similarity(word_list=word_list, use_nan=use_nan)
    pickle.dump(sim, file=open(backup, 'wb'))
    return sim


def _normalize(a):
    return np.interp(a, (np.nanmin(a), np.nanmax(a)), (0, 1))


def get(word_list, normalize=False, force=False, verbose=False):

    word_list = [i.lower() for i in word_list]

    list_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, f"{word_list}"))

    backup = f"{BACKUP_FOLDER}/{list_id}.p"

    if not os.path.exists(backup) or force:
        sim = create(word_list=word_list, backup=backup, use_nan=normalize)

    else:
        sim = pickle.load(file=open(backup, 'rb'))

    if normalize:
        sim = _normalize(sim)

    if verbose:
        for i, j in combinations(range(len(word_list)), r=2):
            a = word_list[i]
            b = word_list[j]
            similarity = sim[i, j]
            print(f"Similarity between {a} and {b} is: {similarity:.2f}")

    return sim
