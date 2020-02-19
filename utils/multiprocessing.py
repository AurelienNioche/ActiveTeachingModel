"""
Adapted from: https://engineering.talentpair.com/django-multiprocessing-153dbcf51dab
"""


import queue
import multiprocessing
import signal

from django import db
from django.conf import settings
from django.core.cache import caches

from tqdm import tqdm


def close_service_connections():
    """ Close all connections before we spawn our processes

    This function should only be used when writing multithreaded scripts where connections need to manually
    opened and closed so that threads don't reuse the same connection

    https://stackoverflow.com/questions/8242837/django-multiprocessing-and-database-connections
    """

    # close db connections, they will be recreated automatically
    db.connections.close_all()

    # close redis connections, will be recreated automatically
    for k in settings.CACHES.keys():
        caches[k].close()


def ignore_keyboard_interrupts():
    """Ignore keyboard interrupt"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def work(func, job_queue, result_queue, stop_event):

    ignore_keyboard_interrupts()

    while not job_queue.empty():
        try:
            i, job = job_queue.get(block=False)
            result_queue.put((i, func(stop_event=stop_event, **job)))
        except queue.Empty:
            pass

    close_service_connections()


class MultiProcess:
    """
     Implemented to be used as a context manager
     (so we dont have to worry about garbage collection calling __del__)
    """

    def __init__(self, n_worker=multiprocessing.cpu_count()):

        self.num_workers = n_worker

        self.job_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()

        self.workers = []

    def __enter__(self):
        close_service_connections()
        return self

    def map(self, func, iterable):

        pbar = tqdm(total=len(iterable))

        for i, it in enumerate(iterable):
            self.job_queue.put((i, it))

        for _ in range(self.num_workers):

            p = multiprocessing.Process(target=work,
                                        args=(
                                            func,
                                            self.job_queue,
                                            self.result_queue,
                                            self.stop_event
                                        ))
            p.start()
            self.workers.append(p)

        n_jobs = len(iterable)
        rv = [None for _ in range(n_jobs)]

        try:

            for _ in range(n_jobs):
                i, r = self.result_queue.get(block=True)
                rv[i] = r
                pbar.update()

            return rv

        except KeyboardInterrupt:
            print("Received Ctr-C")
            self.stop_event.set()
            for p in self.workers:
                p.join()
            exit()
        for p in self.workers:
            p.terminate()
            p.join()

    def __exit__(self, *args, **kwargs):
        pass
