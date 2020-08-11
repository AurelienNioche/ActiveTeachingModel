import os
from tqdm import tqdm
import pickle
import numpy as np
import sys

from task_param.task_param_carlos import TaskParam
from model.teacher.generic import Teacher


SCRIPT_NAME = os.path.basename(__file__).split(".")[0]
PICKLE_FOLDER = os.path.join("pickle", SCRIPT_NAME)

os.makedirs(PICKLE_FOLDER, exist_ok=True)


def run(teacher: Teacher, tk: TaskParam, omniscient: bool):

    n_iter = tk.n_ss * tk.ss_n_iter

    use_teacher_psy = hasattr(teacher, 'psychologist') and not omniscient
    if use_teacher_psy:
        psychologist = teacher.psychologist
        inferred_param = np.zeros((n_iter, len(tk.param)))
    else:
        # You need to create a psychologist to get a probability of recall
        psychologist = tk.psychologist_model.create(tk=tk, omniscient=True)

    np.random.seed(tk.seed)
    hist = np.zeros(n_iter, dtype=int)
    ts = np.zeros(n_iter, dtype=int)  # Unit is second
    success = np.zeros(n_iter, dtype=int)
    # TODO: fill the ts array (ts: timestamp), idem for success
    now = 0

    item = None
    timestamp = None
    was_success = None

    itr = 0

    with tqdm(total=tk.ss_n_iter*tk.n_ss, file=sys.stdout) as pbar:
        for _ in range(tk.n_ss):
            for _ in range(tk.ss_n_iter):

                item = teacher.ask(now=now,
                                   last_was_success=was_success,
                                   last_time_reply=timestamp,
                                   idx_last_q=item)

                timestamp = now
                p = psychologist.p(
                        item=item,
                        param=tk.param,
                        now=timestamp)

                was_success = np.random.random() < p

                # print("itr", itr, "item", item, "p", p, "success", was_success)

                hist[itr] = item

                if use_teacher_psy:
                    inferred_param[itr] = \
                        teacher.psychologist.inferred_learner_param()

                    # TODO: Compute the error of prediction
                    # (average of sum over all the items)
                else:
                    psychologist.learner.update(item=item, timestamp=timestamp)
                    # You don't need to do it here

                now += tk.time_per_iter
                itr += 1
                pbar.update()

            now += tk.time_per_iter * tk.ss_n_iter_between
    if use_teacher_psy:
        return hist, inferred_param

    else:
        return hist


def make_data(tk: TaskParam) -> dict:

    hist = {}
    param_recovery = {}

    for teacher_class, omniscient in zip(tk.teachers, tk.omniscient):

        tqdm.write(f"Simulating '{teacher_class.__name__}'...")
        teacher = teacher_class.create(tk=tk, omniscient=omniscient)
        r = run(teacher=teacher, tk=tk, omniscient=omniscient)

        cond_label = teacher_class.__name__
        if omniscient:
            cond_label += '-omniscient'

        if isinstance(r, tuple):
            hist[cond_label], \
                param_recovery[cond_label] = r
        else:
            hist[cond_label] = r

    return {"kwarg1": tk, "kwarg2": hist}

