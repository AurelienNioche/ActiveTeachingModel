import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "ActiveTeachingModel.settings")
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

from utils.multiprocessing import MultiProcess

from time import sleep
import numpy as np


def f(i, stop_event):
    np.random.seed(i)
    wait = np.random.random()*10
    print(f"process {i} wait {wait} s")
    sleep(wait)
    return i**2


def main():

    with MultiProcess() as mp:

        results = mp.map(func=f, iterable=[{"i": i} for i in range(10)])

    print(results)


if __name__ == "__main__":
    main()
