from django.db import transaction
import threading
import django.db.utils
import psycopg2

import numpy as np
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
            if r == 'n':
                sys.exit()
            elif r == 'y':
                break
            else:
                print("Your response have to be 'y' or 'n'!")
        self.f(**kwargs)
        print("Done!")


import sys


class Tee(object):
    """"
    exemple of usage: sys.stdout = Tee(f'{LOG_DIR}/log.log')
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
