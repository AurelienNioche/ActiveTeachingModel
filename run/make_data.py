import os
from tqdm import tqdm
import pickle
import numpy as np

from teacher.psychologist import Psychologist

SCRIPT_NAME = os.path.basename(__file__).split(".")[0]
PICKLE_FOLDER = os.path.join("pickle", SCRIPT_NAME)

os.makedirs(PICKLE_FOLDER, exist_ok=True)


def run(teacher, tk, omniscient):

    use_teacher_psy = hasattr(teacher, 'psychologist') and not omniscient
    if use_teacher_psy:
        psychologist = teacher.psychologist
    else:
        psychologist = Psychologist.create(tk)

    np.random.seed(tk.seed)
    hist = np.zeros(tk.n_ss * tk.ss_n_iter, dtype=int)
    inferred_param = []

    now = 0

    item = None
    timestamp = None
    was_success = None

    itr = 0

    with tqdm(total=tk.ss_n_iter*tk.n_ss) as pbar:
        for _ in range(tk.n_ss):
            for _ in range(tk.ss_n_iter):
                item = teacher.ask(now=now,
                                   last_was_success=was_success,
                                   last_time_reply=timestamp,
                                   idx_last_q=item)
                timestamp = now
                was_success = np.random.random() < \
                    psychologist.p(
                        item=item,
                        param=tk.param,
                        timestamp=timestamp)
                hist[itr] = item

                if use_teacher_psy:
                    inferred_param.append(
                        teacher.psychologist.inferred_param)
                else:
                    psychologist.update_minimal(item=item, timestamp=timestamp)

                now += tk.time_per_iter
                itr += 1
                pbar.update()

            now += tk.time_per_iter * tk.ss_n_iter_between
    if use_teacher_psy:
        return hist, np.asarray(inferred_param)

    else:
        return hist


def _make_data(tk):

    data = {}
    param_recovery = {}

    for teacher_class, omniscient in zip(tk.teachers, tk.omniscient):

        tqdm.write(f"Simulating '{teacher_class.__name__}'...")
        teacher = teacher_class.create(tk=tk, omniscient=omniscient)

        r = run(teacher=teacher, tk=tk, omniscient=omniscient)

        cond_label = teacher_class.__name__
        if omniscient:
            cond_label += '-omniscient'

        if isinstance(r, tuple):
            data[cond_label], \
                param_recovery[cond_label] = r
        else:
            data[cond_label] = r

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
