import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from model.learner.exponential_n_delta import ExponentialNDelta
# from model.learner.walsh2018 import Walsh2018
from model.psychologist.psychologist_grid import PsychologistGrid
from model.teacher.leitner import Leitner
from model.teacher.recursive import Recursive
from model.teacher.recursive_threshold import RecursiveThreshold
from model.teacher.sampling import Sampling
from model.teacher.threshold import Threshold
from run.make_data_triton import run
from settings.config_triton import (LEARNER, LEARNER_INV, PSY_INV,
                                    PSYCHOLOGIST, TEACHER, TEACHER_INV, Config)
# from utils.cp.grid import generate_param
from utils.convert import dic


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def cp_grid_param(grid_size, bounds, methods):
    """Get grid parameters"""

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

    return grid


def plot_param_space(user_name: str, grid: pd.DataFrame, log_liks: np.ndarray) -> None:
    """Heatmap of the alpha-beta parameter space"""

    print("Plotting heatmap...")
    plt.close()
    log_liks = pd.Series(log_liks, name="log_lik")
    data = pd.concat((grid, log_liks), axis=1)
    try:  # Duplicated entries can appear with rounding
        data = data.round(2).pivot("alpha", "beta", "log_lik")
    except:
        data = data.pivot("alpha", "beta", "log_lik")
    ax = sns.heatmap(data=data, cmap="viridis")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join("fig", f"param_grid_{user_name}.pdf"))
    print("Done!")


def produce_data(data_folder):

    n_item = 150
    omni = True

    learner_md = ExponentialNDelta

    teacher_md = Leitner  # Leitner
    psy_md = PsychologistGrid

    is_item_specific = False

    ss_n_iter = 100
    time_between_ss = 24 * 60 ** 2
    n_ss = 6
    learnt_threshold = 0.9
    time_per_iter = 2

    sampling_cst = {"n_sample": 10000}

    leitner_cst = {"delay_factor": 2, "delay_min": 2}

    pr_lab = ["alpha", "beta"]
    bounds = np.asarray([[0.0000001, 0.00005], [0.0001, 0.9999]])

    gen_grid_size = 20

    psy_pr = {
        "grid_size": 20,
        "grid_methods": [PsychologistGrid.LIN, PsychologistGrid.LIN],
    }
    cst_time = 1

    if teacher_md == Leitner:
        teacher_pr = leitner_cst
    elif teacher_md == Sampling:
        teacher_pr = sampling_cst
    elif teacher_md in (Threshold, Recursive, RecursiveThreshold):
        teacher_pr = {}
    else:
        raise ValueError

    teacher_pr_lab, teacher_pr_val = dic.to_key_val_list(teacher_pr)
    psy_pr_lab, psy_pr_val = dic.to_key_val_list(psy_pr)

    pr_grid = cp_grid_param(
        bounds=bounds,
        grid_size=gen_grid_size,
        methods=np.array([np.linspace, np.linspace]),
    )

    for i, pr_val in tqdm(enumerate(pr_grid), total=len(pr_grid)):

        config_dic = {
            "data_folder": data_folder,
            "config_file": None,
            "seed": 0,
            "agent": i,
            "bounds": bounds,
            "md_learner": LEARNER_INV[learner_md],
            "md_psy": PSY_INV[psy_md],
            "md_teacher": TEACHER_INV[teacher_md],
            "omni": omni,
            "n_item": n_item,
            "is_item_specific": is_item_specific,
            "ss_n_iter": ss_n_iter,
            "time_between_ss": time_between_ss,
            "n_ss": n_ss,
            "learnt_threshold": learnt_threshold,
            "time_per_iter": time_per_iter,
            "cst_time": cst_time,
            "teacher_pr_lab": teacher_pr_lab,
            "teacher_pr_val": teacher_pr_val,
            "psy_pr_lab": psy_pr_lab,
            "psy_pr_val": psy_pr_val,
            "pr_lab": pr_lab,
            "pr_val": pr_val.tolist(),
        }
        config = Config(**config_dic, config_dic=config_dic)

        df = run(config=config, with_tqdm=False)
        f_name = (
            f"{learner_md.__name__}-"
            f"{psy_md.__name__}-"
            f"{teacher_md.__name__}-{i}.csv"
        )
        os.makedirs(config.data_folder, exist_ok=True)
        df.to_csv(os.path.join(config.data_folder, f_name))


def preprocess_data(data_folder, preprocess_data_file):

    assert os.path.exists(data_folder)

    files = [
        p.path
        for p in os.scandir("data/local/explo_leitner")
        if os.path.splitext(p.path)[1] == ".csv"
    ]
    file_count = len(files)
    assert file_count > 0

    row_list = []

    for i, p in tqdm(enumerate(files), total=file_count):
        filename, file_extension = os.path.splitext(p)
        if file_extension != ".csv":
            print(f"ignore {p}")
            continue

        try:
            df = pd.read_csv(p, index_col=[0])

            last_iter = max(df["iter"])
            is_last_iter = df["iter"] == last_iter

            n_learnt = df[is_last_iter]["n_learnt"].iloc[0]

            is_last_ss = df["ss_idx"] == max(df["ss_idx"]) - 1

            ss_n_iter = df["ss_n_iter"][0] - 1

            is_last_iter_ss = df["ss_iter"] == ss_n_iter

            n_learnt_end_ss = df[is_last_ss & is_last_iter_ss]["n_learnt"].iloc[0]

            md_learner = df["md_learner"].iloc[0]
            md_teacher = df["md_teacher"].iloc[0]
            md_psy = df["md_psy"].iloc[0]
            pr_lab = df["pr_lab"].iloc[0]
            pr_val = df["pr_val"].iloc[0]
            pr_lab = eval(pr_lab)
            pr_val = eval(pr_val)
            row = {
                "agent": i,
                "md_learner": md_learner,
                "md_psy": md_psy,
                "md_teacher": md_teacher,
                "n_learnt": n_learnt,
                "n_learnt_end_ss": n_learnt_end_ss,
            }

            for k, v in zip(pr_lab, pr_val):
                row.update({k: v})

            row_list.append(row)

        except Exception as e:
            print(p)
            raise e

    df = pd.DataFrame(row_list)
    df.to_csv(preprocess_data_file)
    return df


def main():
    raw_data_folder = os.path.join("data", "local", "test")
    preprocess_data_file = os.path.join("data", "local", "test.csv")

    force = False

    if not os.path.exists(raw_data_folder) or force:
        produce_data(data_folder=raw_data_folder)

    if not os.path.exists(preprocess_data_file) or force:
        df = preprocess_data(
            data_folder=raw_data_folder, preprocess_data_file=preprocess_data_file
        )
    else:
        df = pd.read_csv(preprocess_data_file, index_col=[0])

    print("Plotting heatmap...")
    # plt.close()
    # log_liks = pd.Series(log_liks, name="log_lik")
    # data = pd.concat((grid, log_liks), axis=1)
    # try:  # Duplicated entries can appear with rounding
    #     data = data.round(2).pivot("alpha", "beta", "log_lik")
    # except:
    #     data = data.pivot("alpha", "beta", "log_lik")
    data = pd.DataFrame(
        {"alpha": df["alpha"], "beta": df["beta"], "n_learnt": df["n_learnt"]}
    )
    data = data.round(8).pivot("alpha", "beta", "n_learnt")
    ax = sns.heatmap(data=data, cmap="viridis", cbar_kws={"label": "N learnt"})
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join("fig", f"param_grid_{user_name}.pdf"))
    print("Done!")


if __name__ == "__main__":

    main()
