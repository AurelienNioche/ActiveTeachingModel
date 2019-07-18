import pickle
import datetime
from time import time
import sys
import os


class Tee(object):
    """"
    example of usage: sys.stdout = Tee(f'{LOG_DIR}/log.log')
    """

    def __init__(self, f):

        self.files = (sys.stdout, open(f, 'w'))

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()

    def __del__(self):
        self.files[-1].close()


def dic2string(dic):
    return str(dic).replace(' ', '_').replace('{', '')\
        .replace('}', '').replace("'", '').replace(':', '').replace(',', '')


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


def print_begin(msg):
    print(msg, end=' ', flush=True)
    return time()


def print_done(t):

    print(f"Done! [time elapsed "
          f"{datetime.timedelta(seconds=time() - t)}]")
