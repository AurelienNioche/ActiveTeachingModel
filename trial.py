import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "ActiveTeachingModel.settings")
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()


import multiprocessing
import os
import signal
import time

from tqdm import tqdm

import numpy as np

import signal
import logging


# def init_worker():
#     signal.signal(signal.SIGINT, signal.SIG_IGN)
#
#
# import multiprocessing, os, signal, time, queue
#
# def do_work(event):
#     print('Work Started: %d' % os.getpid())
#     while True:
#         time.sleep(2)
#         if event.is_set():
#             break
#     return 'Success'
#
# def manual_function(event, job_queue, result_queue):
#     signal.signal(signal.SIGINT, signal.SIG_IGN)
#     while not job_queue.empty():
#         try:
#             job = job_queue.get(block=False)
#             result_queue.put(do_work(event))
#         except queue.Empty:
#             pass
#         #except KeyboardInterrupt: pass
#
#
# def main():
#     job_queue = multiprocessing.Queue()
#     result_queue = multiprocessing.Queue()
#     event = multiprocessing.Event()
#
#     for i in range(6):
#         job_queue.put(None)
#
#     workers = []
#     for i in range(3):
#         tmp = multiprocessing.Process(target=manual_function,
#                                       args=(event, job_queue, result_queue))
#         tmp.start()
#         workers.append(tmp)
#
#     try:
#         for worker in workers:
#             worker.join()
#     except KeyboardInterrupt:
#         print('parent received ctrl-c')
#         event.set()
#         for worker in workers:
#             # worker.terminate()
#             worker.join()
#
#     while not result_queue.empty():
#         print(result_queue.get(block=False))

# if __name__ == "__main__":
#     main()

def run_worker(i, stop_event):

    ## i, e = args
    print(f"start {i}")
    while True:
        multiprocessing.Event().wait(i)
        # print("hey")
        if stop_event.is_set():
            print(" e is set")
            for j in range(100):
                print("yo ", i, j)
            break
        print(f"continue {i}")

    print(f"Done {i}")
    return i

    # print("receive")
    # multiprocessing.Event().wait(2)
    # print("Done")
#
#
# # class MyPool(multiprocessing.Pool):
# #
# #     def __init__(self, *args, **kwargs):
# #         super().__init__(*args, **kwargs)
# #         self.event = multiprocessing.Event()
#
# def main():
#     print("Initializng 5 workers")
#     # pool = multiprocessing.Pool(5)
#
#     print("Starting 3 jobs of 15 seconds each")
#     m = multiprocessing.Manager()
#     p = multiprocessing.Pool(4, init_worker)
#     e = m.Event()
#
#     args = [(i, e) for i in range(4)]
#     p_list = p.imap(run_worker, args)
#     try:
#         try:
#             r = list(tqdm(p_list, total=4))
#             print("Done")
#             p.close()
#         except IndexError:
#             print("index")
#         except KeyboardInterrupt:
#             print("keyboard")
#             e.set()
#
#     except Exception as ex:
#         print(ex)
#         print("key")
#         e.set()
#         multiprocessing.Event().wait()
#         print("Closing")
#         p.close()
#         # p.close()
#         # while True:
#         #     try:
#         #         p.join()
#         #     except ValueError:
#         #         multiprocessing.Event().wait(1)
#         #         print('testing')
#         # p.terminate()
#     finally:
#
#         try:
#             pass
#         except KeyboardInterrupt:
#             print("k")
#
#         e.set()
#         multiprocessing.Event().wait()
#         print("Closing")
#         p.close()
#
#         print("Closing")
#         p.close()
#         print("ta mere")
#         # p.terminate()
#         p.join()
#
#     # time.sleep(1000000)
#
#     # except KeyboardInterrupt:
#     #     print("Caught KeyboardInterrupt, terminating workers")
#     #     pool.terminate()
#     #     pool.join()
#     #
#     # else:
#     print("Quitting normally")
#     #pool.close()
#     # p.join()
#

from utils.multiprocessing import MultiProcess
from django import db

def main():
    db.connections.close_all()

    with MultiProcess() as mp:

        r = mp.map(run_worker, [{"i": i } for i in range(10)])
        print("results", r)


if __name__ == "__main__":
    main()