#!/bin/python3
import json
import os
import shutil

import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
import pandas as pd

from model.learner.exponential import Exponential
from model.psychologist.psychologist_grid import PsyGrid
from model.teacher.leitner import Leitner
from model.teacher.myopic import Myopic
from model.teacher.conservative import Conservative

import settings.paths as paths
from settings.config_triton import LEARNER, PSYCHOLOGIST, TEACHER


TEACHER_INV = {v: k for k, v in TEACHER.items()}
PSY_INV = {v: k for k, v in PSYCHOLOGIST.items()}
LEARNER_INV = {v: k for k, v in LEARNER.items()}


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
        erase = input("Do you want to erase the config folder first? " 
                      "('y' or 'yes')")
        if erase in ("y", "yes"):
            shutil.rmtree(paths.CONFIG_CLUSTER_DIR)
            print("Done!")
        else:
            print("I keep everything as it is")


def delete_data(data_name):
    """Delete cluster data files if user confirms"""

    if os.path.exists(data_name):
        erase = input("Do you want to erase the data folder first? " 
                      "('y' or 'yes')")
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

    os.system(f"/bin/sh config/utils/mod_job.sh " 
              f"{template_path} {config_path} {saving_path}")


def main() -> None:
    """Set the parameters and generate the JSON config files"""

    data_folder = select_data_folder()
    print("Generating cluster config files...")

    # -------------     SET PARAM HERE      ------------------------ #

    learner_md = Exponential
    psy_md = PsyGrid

    teacher_models = (Leitner, Myopic, Conservative)

    ss_n_iter = 100
    time_between_ss = 24 * 60 ** 2
    n_ss = 6
    learnt_threshold = 0.9
    time_per_iter = 4

    n_item = 500

    leitner_cst = {"delay_factor": 2, "delay_min": time_per_iter}

    pr_lab = ["alpha", "beta"]

    bounds = [[2e-07, 0.025], [0.0001, 0.9999]]

    grid_methods = [PsyGrid.GEO, PsyGrid.LIN]
    grid_size = 100

    cst_time = 1

    seed = 123
    np.random.seed(seed)

    n_agent = 100
    df = pd.read_csv("config/parameters/n_learnt_leitner.csv", index_col=0)
    n_learnt = df["n_learnt"].values
    alpha = df["alpha"].values
    beta = df["beta"].values
    smart_enough = np.flatnonzero(n_learnt > 0)
    slc = np.random.choice(smart_enough, size=n_agent, replace=False)
    grid = np.vstack((alpha[slc], beta[slc])).T

    seed = 123
    np.random.seed(seed)
    possibilities = np.vstack((alpha[smart_enough], beta[smart_enough])).T

    grid_spec = np.zeros((n_agent, n_item, len(bounds)))
    for agent in range(n_agent):
        for item in range(n_item):
            idx = np.random.randint(len(possibilities))
            grid_spec[agent, item] = possibilities[idx]

    # ------------------  END OF PARAMETER SETTING -------------------- #

    # ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    pbar = tqdm()
    job_number = 0

    for omni in (True, False):
        for is_item_specific in (False, True):

            for agent in range(n_agent):

                if not is_item_specific:
                    pr_val = grid[agent].tolist()

                else:
                    pr_val = grid_spec[agent].tolist()

                for teacher_md in teacher_models:

                    if teacher_md == Leitner:
                        teacher_pr = leitner_cst
                    elif teacher_md in (Myopic, Conservative):
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
