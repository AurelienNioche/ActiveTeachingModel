import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from model.learner.exponential import Exponential
from model.teacher.leitner import Leitner

from run.make_data_triton import run

from settings.config_triton import Config


def dic_to_key_val_list(dic):
    lab = list(dic.keys())
    val = [dic[k] for k in lab]
    return lab, val


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


def produce_data(raw_data_folder, bounds, methods, grid_size):

    n_item = 150
    omni = True

    learner_md = Exponential
    teacher_md = Leitner

    is_item_specific = False
    ss_n_iter = 100
    time_between_ss = 24 * 60 ** 2
    n_ss = 6
    learnt_threshold = 0.9
    time_per_iter = 4

    pr_lab = ["alpha", "beta"]

    teacher_pr = {"delay_factor": 2, "delay_min": 2}
    teacher_pr_lab, teacher_pr_val = dic_to_key_val_list(teacher_pr)

    pr_grid = cp_grid_param(
        bounds=np.asarray(bounds),
        grid_size=grid_size,
        methods=np.array(methods))

    for i, pr_val in tqdm(enumerate(pr_grid), total=len(pr_grid)):

        config_dic = {
            "data_folder": None,
            "config_file": None,
            "seed": 0,
            "agent": i,
            "bounds": None,
            "md_learner": learner_md.__name__,
            "md_psy": None,
            "md_teacher": teacher_md.__name__,
            "omni": omni,
            "n_item": n_item,
            "is_item_specific": is_item_specific,
            "ss_n_iter": ss_n_iter,
            "time_between_ss": time_between_ss,
            "n_ss": n_ss,
            "learnt_threshold": learnt_threshold,
            "time_per_iter": time_per_iter,
            "cst_time": 1,
            "teacher_pr_lab": teacher_pr_lab,
            "teacher_pr_val": teacher_pr_val,
            "psy_pr_lab": None,
            "psy_pr_val": None,
            "pr_lab": pr_lab,
            "pr_val": pr_val.tolist(),
        }
        config = Config(**config_dic, config_dic=config_dic)

        df = run(config=config, with_tqdm=False)
        df.to_csv(os.path.join(raw_data_folder, f"{i}.csv"))


def preprocess_data(data_folder, preprocess_data_file):

    files = [
        p.path
        for p in os.scandir(data_folder)
        if os.path.splitext(p.path)[1] == ".csv"
    ]
    file_count = len(files)
    assert file_count > 0

    row_list = []

    for i, p in tqdm(enumerate(files), total=file_count):

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
            "n_learnt_end_ss": n_learnt_end_ss}

        for k, v in zip(pr_lab, pr_val):
            row.update({k: v})

        row_list.append(row)

    df = pd.DataFrame(row_list)
    df.to_csv(preprocess_data_file)
    return df


def get_data():

    bounds = [[2e-07, 0.025], [0.0001, 0.9999]]
    methods = [np.geomspace, np.linspace]

    grid_size = 20

    trial_name = str(bounds).replace("[", "]").replace(" ", "_")\
        .replace(",", "").replace(".", "-").replace("]", "") + \
        "_".join([m.__name__ for m in methods])\
        + str(grid_size)
    raw_data_folder = os.path.join("data", "leitner_explo", trial_name)
    os.makedirs(raw_data_folder, exist_ok=True)
    preprocess_folder = os.path.join("data",
                                     "preprocessed",
                                     "explo_leitner")
    os.makedirs(preprocess_folder, exist_ok=True)
    preprocess_data_file = os.path.join(preprocess_folder,
                                        f'{trial_name}.csv')

    force = False

    if not os.path.exists(raw_data_folder) \
            or not [
        p
        for p in os.scandir(raw_data_folder)
        if os.path.splitext(p.path)[1] == ".csv"
            ] or force:
        produce_data(raw_data_folder=raw_data_folder, bounds=bounds,
                     methods=methods, grid_size=grid_size)

    if not os.path.exists(preprocess_data_file) or force:
        df = preprocess_data(
            data_folder=raw_data_folder,
            preprocess_data_file=preprocess_data_file)
    else:
        df = pd.read_csv(preprocess_data_file, index_col=[0])

    data = pd.DataFrame(
        {"alpha": df["alpha"], "beta": df["beta"], "n_learnt": df["n_learnt"]}
    )
    return data


def main():

    data = get_data()

    data_pivoted = data.round(8).pivot("alpha", "beta", "n_learnt")
    ax = sns.heatmap(data=data_pivoted, cmap="viridis",
                     cbar_kws={"label": "N learnt", })
    ax.invert_yaxis()
    plt.tight_layout()
    fig_folder = os.path.join("fig", "explo_leitner")
    os.makedirs(fig_folder, exist_ok=True)
    plt.savefig(os.path.join(fig_folder, f"explo-leitner.png"),
                dpi=300)

    print("Done!")


if __name__ == "__main__":

    main()
