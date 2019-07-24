import sys
import threading

import django.db
import numpy as np
import psycopg2
from django.db import transaction


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