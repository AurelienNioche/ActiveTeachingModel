from django.db import transaction
import threading
import django.db.utils
import psycopg2

import numpy as np
import pickle
import datetime
from time import time
import sys


class Atomic:

    def __init__(self, f):
        self.f = f

    def __call__(self, **kwargs):

        while True:
            try:

                with transaction.atomic():
                    return self.f(**kwargs)

            except (
                    django.db.IntegrityError,
                    django.db.OperationalError,
                    django.db.utils.OperationalError,
                    psycopg2.IntegrityError,
                    psycopg2.OperationalError
            ) as e:
                print("*" * 50)
                print("INTEGRITY ERROR" + "!" * 10)
                print(str(e))
                print("*" * 50)
                threading.Event().wait(1 + np.random.random() * 4)
                continue


class AskUser:

    def __init__(self, f):
        self.f = f

    def __call__(self, **kwargs):

        while True:
            r = input("Are you sure you want to operate this change?")
            r.lower()
            if r in ('n', 'no'):
                sys.exit()
            elif r in ('y', 'yes'):
                break
            else:
                print("Your response has to be 'y' or 'n'!")
        self.f(**kwargs)
        print("Done!")


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

    if verbose:
        t = time()
        print(f"Dumping to file '{file_path}'...", end=' ', flush=True)
    pickle.dump(obj, open(file_path, 'wb'))
    if verbose:
        # noinspection PyUnboundLocalVariable
        print(f"Done! [time elapsed: "
              f"{datetime.timedelta(seconds=time() - t)}]")


def load(file_path, verbose=True):

    if verbose:
        t = time()
        print(f"Loading from file '{file_path}'...", end=' ', flush=True)
    obj = pickle.load(open(file_path, 'rb'))
    if verbose:
        # noinspection PyUnboundLocalVariable
        print(f"Done [time elapsed: "
              f"{datetime.timedelta(seconds=time() - t)}]")
    return obj


def print_begin(msg):
    print(msg, end=' ', flush=True)
    return time()


def print_done(t):

    print(f"Done! [time elapsed "
          f"{datetime.timedelta(seconds=time() - t)}]")
