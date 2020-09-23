#!/bin/python3

import datetime
import json
import os
import shutil

import numpy as np
from numpy.random import default_rng
# from scipy.stats import loguniform
from tqdm import tqdm
import pandas as pd

import settings.paths as paths
from model.learner.exponential_n_delta import ExponentialNDelta
# from model.learner.walsh2018 import Walsh2018
from model.psychologist.psychologist_grid import PsychologistGrid
from model.teacher.leitner import Leitner
from model.teacher.recursive import Recursive
from model.teacher.sampling import Sampling
from model.teacher.threshold import Threshold
from settings.config_triton import LEARNER, PSYCHOLOGIST, TEACHER

TEACHER_INV = {v: k for k, v in TEACHER.items()}
PSY_INV = {v: k for k, v in PSYCHOLOGIST.items()}
LEARNER_INV = {v: k for k, v in LEARNER.items()}


def generate_param(bounds, is_item_specific, n_item):

    param = np.zeros(len(bounds))
    for i, b in enumerate(bounds):
        param[i] = np.random.uniform(b[0], b[1])

    if is_item_specific:

        param_spec = np.zeros((n_item, len(bounds)))
        for i, b in enumerate(bounds):

            # param[:, i] = np.random.uniform(b[0], b[1], n_item)

            v = param[i]
            # std = (b[1] - b[0]) * 0.05
            # v = np.random.normal(loc=mean, scale=std, size=n_item)
            # v[v < b[0]] = b[0]
            # v[v > b[1]] = b[1]

            param_spec[:, i] = v

        param = param_spec

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
    data_folder = os.path.join(paths.DATA_CLUSTER_DIR, trial_name)
    print("Data will be save at:", data_folder)

    cleanup(data_folder)
    return data_folder


def cleanup(data_folder):
    delete_config()
    delete_data(data_folder)

    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(paths.CONFIG_CLUSTER_DIR, exist_ok=True)


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

    data_folder = select_data_folder()
    print("Generating cluster config files...")

    # -------------     SET PARAM HERE      ------------------------ #

    gen_method = "random"

    learner_md = ExponentialNDelta
    psy_md = PsychologistGrid

    teacher_models = (Leitner, Threshold, Recursive)

    ss_n_iter = 100
    time_between_ss = 24 * 60 ** 2
    n_ss = 6
    learnt_threshold = 0.9
    time_per_iter = 4

    n_item = 500  # 500 * n_ss // 6

    sampling_cst = {"n_sample": 10000}

    leitner_cst = {"delay_factor": 2, "delay_min": time_per_iter}

    pr_lab = ["alpha", "beta"]
    ### Good bounds in-silico
    # bounds = [[0.0000001, 0.00005], [0.0001, 0.9999]]
    ### With prior in silico
    bounds = [[0.0000001, 0.025], [0.0001, 0.9999]]
    ### Bounds for user experiment
    # bounds = [[0.0000001, 0.1], [0.0001, 0.9999]]
    grid_methods = [PsychologistGrid.LOG, PsychologistGrid.LIN]
    grid_size = 100  # 20
    # gen_methods = [np.linspace, np.linspace]
    # gen_bounds = [[0.0000001, 0.00005], [0.0001, 0.9999]]
    gen_bounds = [[0.00000273, 0.00005], [0.42106842, 0.9999]]
    cst_time = 1

    seed = 123
    np.random.seed(seed)

    if gen_method == 'random':

        n_agent = 100

    elif gen_method == 'p_depending_on_lls':

        n_agent = 100

        grid_df = pd.read_csv("grid.csv", index_col=[0])
        ll_df = pd.read_csv("lls.csv", index_col=[0])

        lls = np.sum(ll_df.values, axis=1)
        p = (lls - min(lls)) / (max(lls) - min(lls))
        p /= np.sum(p)

        grid = grid_df.values

    elif gen_method == 'best_fit':

        grid = pd.read_csv("best_param.csv", index_col=[0]).values
        n_agent = len(grid)
        print(n_agent)

    elif gen_method == 'use_grid':

        gen_grid_size = 20
        gen_methods = [np.linspace, np.linspace]

        grid = cp_grid_param(
            grid_size=gen_grid_size, methods=gen_methods, bounds=gen_bounds)

        n_agent = len(grid)
    else:
        raise Exception

    # ------------------  END OF PARAMETER SETTING -------------------- #

    # ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    pbar = tqdm()
    job_number = 0

    for omni in (False, ):  # (False, True):
        for is_item_specific in (True, ):  # (False, True):

            for agent in range(n_agent):

                if gen_method == 'use_grid':

                    pr_val = grid[agent]
                    assert not is_item_specific

                elif gen_method == 'random':

                    pr_val = generate_param(
                        bounds=gen_bounds,
                        is_item_specific=is_item_specific,
                        n_item=n_item)

                elif gen_method == 'p_depending_on_lls':

                    idx = np.random.choice(np.arange(len(grid)), p=p)
                    param = grid[idx]

                    if is_item_specific:

                        param_spec = np.zeros((n_item, len(bounds)))
                        for i, b in enumerate(bounds):

                            v = param[i]
                            param_spec[:, i] = v

                        param = param_spec

                    pr_val = param.tolist()

                elif gen_method == 'p_depending_on_best_fit':

                    param = grid[agent]

                    if is_item_specific:

                        param_spec = np.zeros((n_item, len(bounds)))
                        for i, b in enumerate(bounds):
                            v = param[i]
                            param_spec[:, i] = v

                        param = param_spec

                    pr_val = param.tolist()

                else:
                    raise ValueError

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

                    psy_pr_lab = ["grid_size", "grid_methods"]
                    psy_pr_val = [grid_size, grid_methods]

                    learner_md_str = LEARNER_INV[learner_md]
                    psy_md_str = PSY_INV[psy_md]
                    teacher_md_str = TEACHER_INV[teacher_md]

                    spec_str = 'spec' if is_item_specific else 'Nspec'
                    omni_str = 'omni' if omni else 'Nomni'

                    data_folder_run = os.path.join(
                        data_folder,
                        f"{spec_str}-"
                        f"{omni_str}",
                        f"{learner_md_str}-"
                        f"{psy_md_str}-"
                        f"{teacher_md_str}")

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
                        "is_item_specific": is_item_specific,
                        "ss_n_iter": ss_n_iter,
                        "time_between_ss": time_between_ss,
                        "n_ss": n_ss,
                        "learnt_threshold": learnt_threshold,
                        "time_per_iter": time_per_iter,
                        "data_folder": data_folder_run,
                    }

                    f_name = os.path.join(
                        paths.CONFIG_CLUSTER_DIR,
                        f"{job_number}-"
                        f"{learner_md_str}-"
                        f"{psy_md_str}-"
                        f"{teacher_md_str}-"
                        f"a{agent}.json",
                    )

                    with open(f_name, "w") as f:
                        json.dump(json_content, f, sort_keys=False, indent=4)
                    pbar.update()
                    job_number += 1

    pbar.close()
    print("Config files created!")

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
