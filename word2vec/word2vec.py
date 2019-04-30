from gensim.models import KeyedVectors

import os
import numpy as np

from itertools import combinations
from datetime import datetime

import json

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))

BACKUP_FOLDER = f'{SCRIPT_FOLDER}/backup'
DATA_FOLDER = f'{SCRIPT_FOLDER}/data'
PATCH_FOLDER = f'{SCRIPT_FOLDER}/patch'

os.makedirs(BACKUP_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(PATCH_FOLDER, exist_ok=True)

DATA = f'{DATA_FOLDER}/GoogleNews-vectors-negative300.bin'
MODEL = f'{BACKUP_FOLDER}/word_vectors.kv'
REPLACEMENT_DIC = f'{PATCH_FOLDER}/replacement.json'


def _load_model():

    t0 = datetime.utcnow()

    print('Load Word2Vec learner...', end=" ")
    if not os.path.exists(MODEL):
        try:
            model = KeyedVectors.load_word2vec_format(DATA, binary=True)
            model.save(MODEL)
        except Exception:
            raise Exception(f"File '{DATA_FOLDER}/GoogleNews-vectors-negative300.bin' is missing. \n"
                            f"File should be available at "
                            f"'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit'")

    else:
        model = KeyedVectors.load(MODEL, mmap='r')

    t1 = datetime.utcnow()

    print(f"Done in {t1 - t0}")

    return model


def _ask_user_for_replacement(model, word):

    print(f"Warning! Word non existing: '{word}'")

    while True:
        new_word = input("Which word I should use in replacement?")
        if new_word in model.vocab:
            print("Thanks!")
            reply = input(f"Do you confirm the replacement of '{word}' by '{new_word}' (y/n)?")
            if reply in ('y', 'yes', 'ok'):
                print("Replacement registered!")
                return new_word
        else:
            print(f'"{new_word}" doesn\'t exist.')


def _replacement(model, word):

    if os.path.exists(REPLACEMENT_DIC):
        rd = json.load(open(REPLACEMENT_DIC, 'r'))
        if word in rd:
            return rd[word]
    else:
        rd = {}
    new_word = _ask_user_for_replacement(model=model, word=word)
    rd[word] = new_word
    json.dump(obj=rd, fp=open(REPLACEMENT_DIC, 'w'))
    return new_word


def evaluate_similarity(word_list, use_nan=False):

    model = _load_model()

    n_word = len(word_list)

    sim = np.zeros((n_word, n_word))

    if use_nan:
        for i in range(n_word):
            sim[i, i] = np.nan

    for a, b in combinations(word_list, 2):

        i = word_list.index(a)
        j = word_list.index(b)

        if a not in model.vocab:
            a = _replacement(model=model, word=a)
        elif b not in model.vocab:
            b = _replacement(model=model, word=b)

        similarity = model.similarity(a, b)

        sim[i, j] = similarity
        sim[j, i] = similarity

        # print(f"Distance between {a} & {b}: {distance}")

    return sim


def play_with_model():

    model = _load_model()

    while True:

        try:
            a = input("Pick a word")
            assert a in model.vocab
            b = "computer"
            print(f'Similarity between {a} and {b}: {model.similarity(a, b)}')

        except (KeyError, AssertionError):
            print('word not existing')


if __name__ == "__main__":

    play_with_model()
