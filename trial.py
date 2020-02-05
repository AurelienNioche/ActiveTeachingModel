import multiprocessing
import os
import signal
import time

from tqdm import tqdm

import numpy as np

import signal
import logging


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def run_worker(args):

    i, e = args

    while True:
        multiprocessing.Event().wait(3)
        print("hey")
        if e.is_set():
            for j in range(100):
                print("yo ", i, j)
            break

    # print("receive")
    # multiprocessing.Event().wait(2)
    # print("Done")


class MyPool(multiprocessing.Pool):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event = multiprocessing.Event()

def main():
    print("Initializng 5 workers")
    # pool = multiprocessing.Pool(5)

    print("Starting 3 jobs of 15 seconds each")
    # m = multiprocessing.Manager()
    p = MyPool(4, init_worker)
    e = p.event

    args = [(i, e) for i in range(4)]

    try:
        r = list(tqdm(p.imap(run_worker, args), total=4))
        p.close()
    except KeyboardInterrupt:
        e.set()
        multiprocessing.Event().wait()
        p.close()
        # p.close()
        # while True:
        #     try:
        #         p.join()
        #     except ValueError:
        #         multiprocessing.Event().wait(1)
        #         print('testing')
        # p.terminate()
    finally:
        p.close()
        print("ta mere")
        # p.terminate()
        p.join()

    # time.sleep(1000000)

    # except KeyboardInterrupt:
    #     print("Caught KeyboardInterrupt, terminating workers")
    #     pool.terminate()
    #     pool.join()
    #
    # else:
    print("Quitting normally")
    #pool.close()
    # p.join()

if __name__ == "__main__":
    main()