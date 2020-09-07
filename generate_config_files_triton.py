#!/bin/python3

import datetime
import json
import os
import shutil

import numpy as np
from numpy.random import default_rng
from scipy.stats import loguniform
from tqdm import tqdm

import settings.paths as paths
from model.learner.exponential_n_delta import ExponentialNDelta
from model.learner.walsh2018 import Walsh2018
from model.psychologist.psychologist_grid import PsychologistGrid
from model.teacher.leitner import Leitner
from model.teacher.recursive import Recursive
from model.teacher.sampling import Sampling
from model.teacher.threshold import Threshold
from settings.config_triton import LEARNER, PSYCHOLOGIST, TEACHER

N_SEC_PER_DAY = 86400
N_SEC_PER_ITER = 2

TEACHER_INV = {v: k for k, v in TEACHER.items()}
PSY_INV = {v: k for k, v in PSYCHOLOGIST.items()}
LEARNER_INV = {v: k for k, v in LEARNER.items()}


def generate_param(bounds, methods, is_item_specific, n_item):

    if is_item_specific:
        param = np.zeros((n_item, len(bounds)))
        for i, b in enumerate(bounds):
            param[:, i] = methods[i](b[0], b[1], n_item)
            # mean = np.random.uniform(b[0], b[1], size=n_item)
            # std = (b[1] - b[0]) * 0.1
            # v = np.random.normal(loc=mean, scale=std, size=n_item)
            # v[v < b[0]] = b[0]
            # v[v > b[1]] = b[1]
            # param[:, i] = v

    else:
        param = np.zeros(len(bounds))
        for i, b in enumerate(bounds):
            param[i] = methods[i](b[0], b[1])

    param = param.tolist()
    return param


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def cp_grid_param(grid_size, bounds, methods):
    """Get grid parameters"""

    bounds = np.asarray(bounds)
    methods = np.asarray(methods)

    diff = bounds[:, 1] - bounds[:, 0] > 0
    not_diff = np.invert(diff)

    values = np.atleast_2d(
        [m(*b, num=grid_size) for (b, m) in zip(bounds[diff], methods[diff])]
    )
    var = cartesian_product(*values)
    grid = np.zeros((max(1, len(var)), len(bounds)))
    if np.sum(diff):
        grid[:, diff] = var
    if np.sum(not_diff):
        grid[:, not_diff] = bounds[not_diff, 0]

    return grid.tolist()


def dic_to_lab_val(dic):
    lab = list(dic.keys())
    val = [dic[k] for k in dic]
    return lab, val


def select_data_folder():
    """Name file, keep last (this) name"""

    trial_name = input("Trial name: ")
    saving_path = os.path.join(paths.DATA_CLUSTER_DIR, trial_name)
    print("Data will be save at:", saving_path)
    return saving_path


def delete_config():
    """Delete cluster config files if user confirms"""

    if os.path.exists(paths.CONFIG_CLUSTER_DIR):
        erase = input("Do you want to erase the config folder first? " "('y' or 'yes')")
        if erase in ("y", "yes"):
            shutil.rmtree(paths.CONFIG_CLUSTER_DIR)
            print("Done!")
        else:
            print("I keep everything as it is")


def delete_data(data_name):
    """Delete cluster data files if user confirms"""

    if os.path.exists(data_name):
        erase = input("Do you want to erase the data folder first? " "('y' or 'yes')")
        if erase in ("y", "yes"):
            shutil.rmtree(data_name)
            print("Done!")
        else:
            print("I keep everything as it is")


def cleanup(data_name):
    delete_config()
    delete_data(data_name)

    os.makedirs(data_name, exist_ok=True)
    os.makedirs(paths.CONFIG_CLUSTER_DIR, exist_ok=True)


def run_simulation():
    simulate = input("Run simulation? " "('y' or 'yes')")
    if simulate in ("y", "yes"):
        os.system("/bin/sh run.sh simulation.job")
    else:
        print("Nothing run.")


def mod_job_file(template_path: str, config_path: str, saving_path: str) -> None:
    """Call the shell script that modifies the number of parallel branches"""

    os.system(f"/bin/sh mod_job.sh " f"{template_path} {config_path} {saving_path}")


def main() -> None:
    """Set the parameters and generate the JSON config files"""

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    learner_models = (ExponentialNDelta,)
    teacher_models = (Leitner, Threshold, Recursive)
    psy_models = (PsychologistGrid,)

    data_folder = select_data_folder()
    cleanup(data_folder)

    seed = 123
    np.random.seed(seed)

    # n_agent = 100
    n_item = 500
    omni = False

    is_item_specific = False

    ss_n_iter = 100
    time_between_ss = 24 * 60 ** 2
    n_ss = 6
    learnt_threshold = 0.9
    time_per_iter = 4

    sampling_cst = {"n_sample": 10000}

    leitner_cst = {"delay_factor": 2, "delay_min": time_per_iter}

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
        "bounds": [[0.0000001, 0.00005], [0.0001, 0.9999]],
        "grid_methods": [PsychologistGrid.LIN, PsychologistGrid.LIN],
        "grid_size": 20,
        "gen_methods": [np.linspace, np.linspace],
        "gen_bounds": [[0.0000001, 0.00005], [0.0001, 0.9999]],
        "cst_time": 1,  # 1 / (60 ** 2),  # 1 / (24 * 60**2),
    }

    job_number = 0

    print("Generating cluster config files...")

    # n_job = len(learner_models) * len(teacher_models) \
    #     * len(psy_models) * n_agent
    pbar = tqdm()  # total=n_job)

    for learner_md in learner_models:

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
        gen_methods = cst["gen_methods"]
        gen_bounds = cst["gen_bounds"]

        grid = cp_grid_param(
            grid_size=grid_size, methods=gen_methods, bounds=gen_bounds
        )

        n_agent = len(grid)
        for agent in range(n_agent):

            pr_val = grid[agent]

            # pr_val = generate_param(
            #     bounds=gen_bounds,
            #     is_item_specific=is_item_specific,
            #     n_item=n_item,
            #     methods=gen_methods)

            for teacher_md in teacher_models:

                if teacher_md == Leitner:
                    teacher_pr = leitner_cst
                elif teacher_md == Sampling:
                    teacher_pr = sampling_cst
                elif teacher_md in (Threshold, Recursive):
                    teacher_pr = {}
                else:
                    raise ValueError

                teacher_pr_lab, teacher_pr_val = dic_to_lab_val(teacher_pr)

                for psy_md in psy_models:

                    if psy_md == PsychologistGrid:
                        psy_pr_lab = ["grid_size", "grid_methods"]
                        psy_pr_val = [grid_size, grid_methods]
                    else:
                        raise ValueError

                    learner_md_str = LEARNER_INV[learner_md]
                    psy_md_str = PSY_INV[psy_md]
                    teacher_md_str = TEACHER_INV[teacher_md]

                    data_folder_run = os.path.join(
                        data_folder,
                        f"{learner_md_str}-" f"{psy_md_str}-" f"{teacher_md_str}",
                    )

                    json_content = {
                        "seed": seed + agent,
                        "agent": agent,
                        "bounds": bounds,
                        "md_learner": learner_md_str,
                        "md_psy": psy_md_str,
                        "md_teacher": teacher_md_str,
                        "omni": omni,
                        "n_item": n_item,
                        "teacher_pr_lab": teacher_pr_lab,
                        "teacher_pr_val": teacher_pr_val,
                        "psy_pr_lab": psy_pr_lab,
                        "psy_pr_val": psy_pr_val,
                        "pr_lab": pr_lab,
                        "pr_val": pr_val,
                        "cst_time": cst_time,
                        "data_folder": data_folder_run,
                        "is_item_specific": is_item_specific,
                        "ss_n_iter": ss_n_iter,
                        "time_between_ss": time_between_ss,
                        "n_ss": n_ss,
                        "learnt_threshold": learnt_threshold,
                        "time_per_iter": time_per_iter,
                    }

                    f_name = os.path.join(
                        paths.CONFIG_CLUSTER_DIR,
                        f"{job_number}-{ts}-"
                        f"{learner_md_str}-"
                        f"{psy_md_str}-"
                        f"{teacher_md_str}-"
                        f"{agent}.json",
                    )

                    with open(f_name, "w") as f:
                        json.dump(json_content, f, sort_keys=False, indent=4)
                    pbar.update()
                    job_number += 1
    pbar.close()
    print("Config files created!")

    os.makedirs(paths.DATA_CLUSTER_DIR, exist_ok=True)
    os.makedirs(paths.LOG_CLUSTER_DIR, exist_ok=True)

    mod_job_file(
        os.path.join(paths.TEMPLATE_DIR, "template.job"),
        paths.CONFIG_CLUSTER_DIR,
        os.path.join(paths.BASE_DIR, "simulation.job"),
    )

    run_simulation()

    # SBATCH --mail-type=END
    # SBATCH --mail-user=nioche.aurelien@gmail.com


if __name__ == "__main__":
    main()
