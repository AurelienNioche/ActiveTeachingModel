"""
Run simulations and save results
"""

from settings.config_triton import Config
from run.make_data_triton import run

import os
import numpy as np

from model.learner.exponential_n_delta import ExponentialNDelta
from model.learner.walsh2018 import Walsh2018
from model.psychologist.psychologist_grid import PsychologistGrid
from model.teacher.leitner import Leitner
from model.teacher.sampling import Sampling
from model.teacher.threshold import Threshold
from model.teacher.recursive import Recursive
from model.teacher.recursive_threshold import RecursiveThreshold

from settings.config_triton import LEARNER, PSYCHOLOGIST, TEACHER

N_SEC_PER_DAY = 24 * 60**2  # 86400
N_SEC_PER_ITER = 2

TEACHER_INV = {v: k for k, v in TEACHER.items()}
PSY_INV = {v: k for k, v in PSYCHOLOGIST.items()}
LEARNER_INV = {v: k for k, v in LEARNER.items()}


def dic_to_lab_val(dic):
    lab = list(dic.keys())
    val = [dic[k] for k in dic]
    return lab, val


def main():

    n_item = 150
    omni = True

    learner_md = ExponentialNDelta
    pr_val = [[0.216, 0.44] for i in range(n_item)]

    teacher_md = Recursive  # Leitner
    psy_md = PsychologistGrid

    is_item_specific = len(np.asarray(pr_val).shape) > 1

    ss_n_iter = 100
    time_between_ss = 86200
    n_ss = 6
    learnt_threshold = 0.9
    time_per_iter = 2

    sampling_cst = {"n_sample": 10000}

    leitner_cst = {"delay_factor": 2, "delay_min": 2}

    walsh_cst = {
        "param_labels": ["tau", "s", "b", "m", "c", "x"],
        "bounds": [
            [0.5, 1.5],
            [0.001, 0.10],
            [0.001, 0.2],
            [0.001, 0.2],
            [0.1, 0.1],
            [0.6, 0.6],
        ],
        "grid_methods": [PsychologistGrid.LIN for _ in range(6)],
        "grid_size": 10,
        "cst_time": 1 / (4 * 24 * 60 ** 2),
    }

    exp_decay_cst = {
        "param_labels": ["alpha", "beta"],
        "bounds": [[0.001, 0.5], [0.00, 0.5]],
        "grid_methods": [PsychologistGrid.LOG, PsychologistGrid.LIN],
        "grid_size": 20,
        "cst_time": 1 / (60 ** 2),  # 1 / (24 * 60**2),
    }

    if learner_md == Walsh2018:
        cst = walsh_cst
    elif learner_md == ExponentialNDelta:
        cst = exp_decay_cst
    else:
        raise ValueError

    bounds = cst["bounds"]
    grid_size = cst["grid_size"]
    grid_methods = cst["grid_methods"]
    pr_lab = cst["param_labels"]
    cst_time = cst["cst_time"]

    if teacher_md == Leitner:
        teacher_pr = leitner_cst
    elif teacher_md == Sampling:
        teacher_pr = sampling_cst
    elif teacher_md in (Threshold, Recursive, RecursiveThreshold):
        teacher_pr = {}
    else:
        raise ValueError

    teacher_pr_lab, teacher_pr_val = dic_to_lab_val(teacher_pr)

    if psy_md == PsychologistGrid:
        psy_pr_lab = [
            "grid_size", "grid_methods"
        ]
        psy_pr_val = [
            grid_size, grid_methods
        ]
    else:
        raise ValueError

    config = Config(
        data_folder="data/local",
        config_file=None,
        config_dic={},
        seed=0,
        agent=0,
        bounds=bounds,
        md_learner=LEARNER_INV[learner_md],
        md_psy=PSY_INV[psy_md],
        md_teacher=TEACHER_INV[teacher_md],
        omni=omni,
        n_item=n_item,
        is_item_specific=is_item_specific,
        ss_n_iter=ss_n_iter,
        time_between_ss=time_between_ss,
        n_ss=n_ss,
        learnt_threshold=learnt_threshold,
        time_per_iter=time_per_iter,
        cst_time=cst_time,
        teacher_pr_lab=teacher_pr_lab,
        teacher_pr_val=teacher_pr_val,
        psy_pr_lab=psy_pr_lab,
        psy_pr_val=psy_pr_val,
        pr_lab=pr_lab,
        pr_val=pr_val,
    )
    df = run(config=config, with_tqdm=True)
    f_name = f"{learner_md.__name__}-" \
             f"{psy_md.__name__}-" \
             f"{teacher_md.__name__}.csv"
    os.makedirs(config.data_folder, exist_ok=True)
    df.to_csv(os.path.join(config.data_folder, f_name))


if __name__ == "__main__":
    main()

