import os
import pickle
import numpy as np

BKP_FOLDER = os.path.join("bkp", "run")


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

    def call_func(*args, **kwargs):

        os.makedirs(os.path.join(BKP_FOLDER, f"{func.__name__}"),
                    exist_ok=True)

        idx_file = file_name('idx')

        info = {k: v for k, v in kwargs.items()}
        info.update({'args': args})

        if os.path.exists(idx_file):

            idx = load(idx_file)
            for i in range(idx+1):

                info_loaded = load(file_name(f"{i}_info"))

                #  Compare 'info' and 'info_loaded'...
                same = True
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

                # ...if they are the sum, load from the associated datafile
                if same:
                    data = load(file_name(f"{i}_data"))
                    return data

        else:
            idx = -1

        idx += 1

        data = func(*args, **kwargs)

        data_file = file_name(f"{idx}_data")
        info_file = file_name(f"{idx}_info")

        dump(data, data_file)
        dump(info, info_file)
        dump(idx, idx_file)

        return data

    return call_func
