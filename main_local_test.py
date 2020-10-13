"""
Run simulations and save results
"""

import os
import numpy as np

from settings.config_triton import Config
from run.make_data_triton import run

from model.learner.exponential import Exponential
from model.learner.walsh2018 import Walsh2018
from model.psychologist.psychologist_grid import PsyGrid
from model.teacher.leitner import Leitner
from model.teacher.myopic import Myopic
from model.teacher.conservative import Conservative
from model.teacher.robust import Robust


def dic_to_lab_val(dic):
    lab = list(dic.keys())
    val = [dic[k] for k in dic]
    return lab, val


def main():

    n_item = 500
    omni = False

    learner_md = Exponential
    pr_val = [[2e-05, 0.5]
              for _ in range(n_item)]

    is_item_specific = len(np.asarray(pr_val).shape) > 1

    teacher_md = Robust
    psy_md = PsyGrid

    ss_n_iter = 100
    time_between_ss = 24 * 60**2
    n_ss = 6
    learnt_threshold = 0.9
    time_per_iter = 4

    leitner_cst = {"delay_factor": 2, "delay_min": 4}

    pr_lab = ["alpha", "beta"],
    bounds = [[2e-07, 0.025], [0.0001, 0.9999]]

    grid_methods = [PsyGrid.GEO, PsyGrid.LIN]
    grid_size = 100
    cst_time = 1

    assert learner_md == Exponential

    if teacher_md == Leitner:
        teacher_pr = leitner_cst
    elif teacher_md in (Myopic, Conservative, Robust):
        teacher_pr = {}
    else:
        raise ValueError

    teacher_pr_lab, teacher_pr_val = dic_to_lab_val(teacher_pr)

    assert psy_md == PsyGrid

    psy_pr_lab = [
        "grid_size", "grid_methods"
    ]
    psy_pr_val = [
        grid_size, grid_methods
    ]

    config = Config(
        data_folder="data/local",
        config_file=None,
        config_dic={},
        seed=0,
        agent=0,
        bounds=bounds,
        md_learner=learner_md.__name__,
        md_psy=psy_md.__name__,
        md_teacher=teacher_md.__name__,
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

