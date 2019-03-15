import os
import pickle
import uuid

from word2vec import word2vec

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
BACKUP_FOLDER = f'{SCRIPT_FOLDER}/backup'

os.makedirs(BACKUP_FOLDER, exist_ok=True)


def create(word_list, backup):

    sim = word2vec.evaluate_similarity(word_list)
    pickle.dump(sim, file=open(backup, 'wb'))
    return sim


def get(word_list):

    list_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, f"{word_list}"))

    backup = f"{BACKUP_FOLDER}/{list_id}.p"

    if not os.path.exists(backup):
        sim = create(word_list=word_list, backup=backup)

    else:
        sim = pickle.load(file=open(backup, 'rb'))

    return sim
