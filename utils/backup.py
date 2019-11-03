import datetime
import os
import pickle
from time import time


def dump(obj, file_path, verbose=True):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if verbose:
        t = time()
        print(f"Dumping to file '{file_path}'...", end=' ', flush=True)
    pickle.dump(obj, open(file_path, 'wb'))
    if verbose:
        # noinspection PyUnboundLocalVariable
        print(f"Done! [time elapsed: "
              f"{datetime.timedelta(seconds=time() - t)}]")


def load(file_path, verbose=True):

    if not os.path.exists(file_path):
        if verbose:
            print(f"No backup file '{file_path}' existing.")
        return None

    if verbose:
        t = time()
        print(f"Loading from file '{file_path}'...", end=' ', flush=True)
    obj = pickle.load(open(file_path, 'rb'))
    if verbose:
        # noinspection PyUnboundLocalVariable
        print(f"Done! [time elapsed: "
              f"{datetime.timedelta(seconds=time() - t)}]\n")
    return obj
