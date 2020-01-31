import os
import pickle
import numpy as np
import glob

from model.settings import BKP_FOLDER


def use_pickle(func):

    """
    Decorator that does the following:
    * If a pickle file corresponding to the call exists,
    it load the data from it instead of calling the function.
    * If no such pickle file exists, it calls 'func',
    creates the file and saves the output in it
    :param func: any function
    :return: output of func(*args, **kwargs)
    """

    def file_name(suffix):
        return os.path.join(BKP_FOLDER, f"{func.__name__}", f"{suffix}.p")

    def load(f_name):
        return pickle.load(open(f_name, 'rb'))

    def dump(obj, f_name):
        return pickle.dump(obj, open(f_name, 'wb'))

    def extract_id(f_name):
        return int("".join([s for s in
                            os.path.basename(f_name) if s.isdigit()]))

    def call_func(lock=None, *args, **kwargs):

        data = None

        os.makedirs(os.path.join(BKP_FOLDER, f"{func.__name__}"),
                    exist_ok=True)

        info = {k: v for k, v in kwargs.items()}
        info.update({'args': args})

        info_files = glob.glob(file_name('*_info'))

        for info_file in info_files:

            info_loaded = load(info_file)

            #  Compare 'info' and 'info_loaded'...
            same = True

            if len(info_loaded) != len(info):
                same = False

            else:
                for k in info_loaded.keys():
                    try:
                        if not info_loaded[k] == info[k]:
                            same = False
                            break

                    # Comparison term to term for arrays
                    except ValueError:
                        if not np.all([info_loaded[k][i] == info[k][i]
                                       for i in range(len(info[k]))]):
                            same = False
                            break
                    except KeyError:
                        same = False
                        break
            # ...if they are the sum, load from the associated datafile
            if same:
                idx = extract_id(info_file)
                data = load(file_name(f"{idx}_data"))
                break

        if data is None:
            data = func(*args, **kwargs)

            if lock is not None:
                lock.acquire()

            info_files = glob.glob(file_name('*_info'))
            if info_files:
                idx = max([extract_id(f_name) for f_name in info_files]) \
                      + 1

            else:
                idx = 0

            data_file = file_name(f"{idx}_data")
            info_file = file_name(f"{idx}_info")

            dump(data, data_file)
            dump(info, info_file)

            if lock is not None:
                lock.release()

        return data

    return call_func
