import os
import pickle

from user_data.settings import BKP_USER_LIST


def get(force=False):

    if not force and os.path.exists(BKP_USER_LIST):
        user_id = pickle.load(open(BKP_USER_LIST, 'rb'))
    else:
        raise Exception("Data not found")

    return user_id
