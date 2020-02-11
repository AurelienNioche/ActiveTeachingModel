import os
import pickle

from user_data.settings import BKP_DATA


def get():

    if os.path.exists(BKP_DATA):
        data = pickle.load(open(BKP_DATA, 'rb'))
    else:
        raise Exception("Data not found")

    return data
