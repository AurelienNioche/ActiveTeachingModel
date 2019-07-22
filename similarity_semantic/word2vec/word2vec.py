import os
import numpy as np

from itertools import combinations
from datetime import datetime

import json

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))

BACKUP_FOLDER = os.path.join('bkp', 'similarity_semantic', 'word2vec')
DATA_FOLDER = os.path.join('data', 'word2vec')

os.makedirs(BACKUP_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

DATA = os.path.join(DATA_FOLDER, 'GoogleNews-vectors-negative300.bin')
MODEL = os.path.join(BACKUP_FOLDER, 'word_vectors.kv')
REPLACEMENT_DIC = os.path.join(DATA_FOLDER, 'replacement.json')


def _load_model():

    from gensim.models import KeyedVectors

    t0 = datetime.utcnow()

    print('Load Word2Vec data...', end=" ", flush=True)
    if not os.path.exists(MODEL):
        try:
            print('Loading from the "bin" file, it can take a while...',
                  end=" ", flush=True)
            model = KeyedVectors.load_word2vec_format(DATA, binary=True)
            model.save(MODEL)
        except Exception:
            raise Exception(
                f"File '{DATA_FOLDER}/GoogleNews-vectors-negative300.bin'"
                f" is missing. \n"
                f"File should be available at "
                f"'https://drive.google.com/"
                f"file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit'")

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
            reply = input(f"Do you confirm the replacement of '{word}' "
                          f"by '{new_word}' (y/n)?")
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

        if b not in model.vocab:
            b = _replacement(model=model, word=b)

        raw_similarity = model.similarity(a, b)

        sim[i, j] = np.abs(raw_similarity)
        sim[j, i] = np.abs(raw_similarity)

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
