import datetime
import sys
from time import time


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


def print_begin(msg):
    print(msg, end=' ', flush=True)
    return time()


def print_done(t):

    print(f"Done! [time elapsed "
          f"{datetime.timedelta(seconds=time() - t)}]")