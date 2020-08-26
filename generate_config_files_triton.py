#%%
import datetime
import os
import json

import numpy as np

from numpy.random import default_rng
from tqdm import tqdm

import settings.paths as paths

from model.teacher.sampling import Sampling
from model.teacher.threshold import Threshold
from model.teacher.leitner import Leitner
from model.learner.exponential_n_delta import ExponentialNDelta
from model.learner.walsh2018 import Walsh2018
from model.psychologist.psychologist_grid import PsychologistGrid

from settings.config_triton import TEACHER, PSYCHOLOGIST, LEARNER


N_SEC_PER_DAY = 86400
N_SEC_PER_ITER = 2

# FOLDER = os.path.join("config", "triton")

# os.makedirs(FOLDER, exist_ok=True)

TEACHER_INV = {v: k for k, v in TEACHER.items()}
PSY_INV = {v: k for k, v in PSYCHOLOGIST.items()}
LEARNER_INV = {v: k for k, v in LEARNER.items()}


def generate_param(bounds, is_item_specific, n_item):

    if is_item_specific:
        param = np.zeros((n_item, len(bounds)))
        for i, b in enumerate(bounds):
            mean = np.random.uniform(b[0], b[1], size=n_item)
            std = (b[1] - b[0]) * 0.1
            v = np.random.normal(loc=mean, scale=std, size=n_item)
            v[v < b[0]] = b[0]
            v[v < [b[1]]] = b[1]
            param[:, i] = v

    else:
        param = np.zeros(len(bounds))
        for i, b in enumerate(bounds):
            param[i] = np.random.uniform(b[0], b[1])

    param = param.tolist()
    return param


def dic_to_lab_val(dic):
    lab = list(dic.keys())
    val = [dic[k] for k in dic]
    return lab, val


def main() -> None:
    """Set the parameters and generate the JSON config files"""

    np.random.seed(1234)

    n_agent = 100
    n_item = 100
    omni = False

    task_param = {
        "is_item_specific": False,
        "ss_n_iter": 100,
        "time_between_ss": 86200,
        "n_ss": 15,
        "learnt_threshold": 0.9,
        "time_per_iter": 2,
    }

    task_pr_lab, task_pr_val = dic_to_lab_val(task_param)

    sampling_cst = {"n_sample": 500}

    leitner_cst = {"delay_factor": 2, "delay_min": 2}

    walsh_cst = {
        "param_labels": ["tau", "s", "b", "m", "c", "x"],
        "bounds": [
            [0.5, 1.5],
            [0.005, 0.10],
            [0.005, 0.2],
            [0.005, 0.2],
            [0.1, 0.1],
            [0.6, 0.6],
        ],
        "grid_size": 10,
    }

    exp_decay_cst = {
        "param_labels": ["alpha", "beta"],
        "bounds": [[0.001, 0.2], [0.00, 0.5]],
        "grid_size": 20,
    }

    learner_models = (ExponentialNDelta,)
    teacher_models = (Leitner, Sampling, Threshold)
    psy_models = (PsychologistGrid,)

    # f_name_root = f"at-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}"
    f_name_root = f"at-{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')}"

    for learner_md in learner_models:

        if learner_md == Walsh2018:
            cst = walsh_cst
        elif learner_md == ExponentialNDelta:
            cst = exp_decay_cst
        else:
            raise ValueError

        bounds = cst["bounds"]
        grid_size = cst["grid_size"]
        pr_lab = cst["param_labels"]

        print("Generating cluster config files...")
        for agent in tqdm(range(n_agent)):

            for teacher_md in teacher_models:

                if teacher_md == Leitner:
                    teacher_pr = leitner_cst
                elif teacher_md == Sampling:
                    teacher_pr = sampling_cst
                elif teacher_md == Threshold:
                    teacher_pr = {}
                else:
                    raise ValueError

                teacher_pr_lab, teacher_pr_val = dic_to_lab_val(teacher_pr)

                for psy_md in psy_models:

                    if psy_md == PsychologistGrid:
                        psy_pr_lab = [
                            "grid_size",
                        ]
                        psy_pr_val = [
                            grid_size,
                        ]
                    else:
                        raise ValueError

                    pr_val = generate_param(
                        bounds=bounds,
                        is_item_specific=task_param["is_item_specific"],
                        n_item=n_item,
                    )

                    json_content = {
                        "seed": agent,
                        "agent": agent,
                        "bounds": bounds,
                        "md_learner": LEARNER_INV[learner_md],
                        "md_psy": PSY_INV[psy_md],
                        "md_teacher": TEACHER_INV[teacher_md],
                        "omni": omni,
                        "n_item": n_item,
                        "task_pr_lab": task_pr_lab,
                        "task_pr_val": task_pr_val,
                        "teacher_pr_lab": teacher_pr_lab,
                        "teacher_pr_val": teacher_pr_val,
                        "psy_pr_lab": psy_pr_lab,
                        "psy_pr_val": psy_pr_val,
                        "pr_lab": pr_lab,
                        "pr_val": pr_val,
                    }

                    # f_name = os.path.join(FOLDER, f"{f_name_root}-{agent}.json")
                    f_name = os.path.join(
                        paths.CONFIG_CLUSTER_DIR,
                        f"{f_name_root}-{json_content['md_learner']}-{json_content['md_psy']}-{json_content['md_teacher']}-{agent}.json",
                    )
                    with open(f_name, "w") as f:
                        json.dump(json_content, f, sort_keys=False, indent=4)
    print("Done!")


if __name__ == "__main__":
    main()