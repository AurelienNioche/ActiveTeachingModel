import os
from tqdm import tqdm
import pickle

SCRIPT_NAME = os.path.basename(__file__).split(".")[0]
PICKLE_FOLDER = os.path.join("pickle", SCRIPT_NAME)

os.makedirs(PICKLE_FOLDER, exist_ok=True)


def _make_data(tk):

    data = {}
    param_recovery = {}

    for teacher_class in tk.teachers:

        tqdm.write(f"Simulating '{teacher_class.__name__}'...")
        r = teacher_class.run(tk)
        if isinstance(r, tuple):
            data[teacher_class.__name__], \
                param_recovery[teacher_class.__name__] = r
        else:
            data[teacher_class.__name__] = r

    return {"tk": tk, "data": data, "param_recovery": param_recovery}


def make_data(tk, force=False):

    bkp_file = os.path.join(PICKLE_FOLDER, f"{tk.extension}.p")

    if os.path.exists(bkp_file) and not force:
        with open(bkp_file, 'rb') as f:
            data = pickle.load(f)

    else:
        data = _make_data(tk=tk)
        with open(bkp_file, 'wb') as f:
            pickle.dump(data, f)

    return data
