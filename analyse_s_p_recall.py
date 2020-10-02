import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(raw_data_folder, preprocess_file):

    assert os.path.exists(raw_data_folder)

    path, dirs, files = next(os.walk(raw_data_folder))
    file_count = len(files)

    assert file_count > 0

    row_list = []

    for i, p in tqdm(enumerate(os.scandir(raw_data_folder)), total=file_count):
        df = pd.read_csv(p.path, index_col=[0])
        df.sort_values("iter", inplace=True)

        for t, p_err in enumerate(df["p_err_mean"]):

            if np.isnan(p_err):
                continue

            row = {
                "agent": i,
                # "Learner": df["md_learner"][0],
                # "Psychologist": df["md_psy"][0],
                # "Teacher": df["md_teacher"][0],
                "p_err_mean": p_err,
                "time": t,
            }
            row_list.append(row)

    df = pd.DataFrame(row_list)
    df.to_csv(preprocess_file)
    return df


def get_data(trial_name, condition_name, teacher_name, force=False):
    teacher_folder = f"exp_decay-psy_grid-{teacher_name}"
    raw_data_folder = os.path.join("data", "triton",
                                   trial_name, condition_name,
                                   teacher_folder)
    assert os.path.exists(raw_data_folder)

    preprocessed_folder = os.path.join("data", "preprocessed",
                                       "analyse_p_recall")

    preprocess_file = os.path.join(
        preprocessed_folder,
        f"{trial_name}_{condition_name}_{teacher_name}_p_recall.csv")

    os.makedirs(preprocessed_folder, exist_ok=True)

    if not os.path.exists(preprocess_file) or force:
        df = preprocess_data(
            raw_data_folder=raw_data_folder,
            preprocess_file=preprocess_file)
    else:
        df = pd.read_csv(preprocess_file, index_col=[0])

    return df


def main(force=False):

    trial_name = "explo_leitner_geolin_rnditem"  # input("Trial name: ")
    for condition_name in "Nspec-Nomni", "spec-Nomni":

        for teacher_name in "forward", "threshold":

            data = get_data(trial_name=trial_name,
                            condition_name=condition_name,
                            teacher_name=teacher_name,
                            force=force)

            fig_folder = os.path.join("fig", trial_name)
            os.makedirs(fig_folder, exist_ok=True)

            fig_path = os.path.join(
                fig_folder,
                f"{condition_name}_{teacher_name}_p_recall.png")

            fig, ax = plt.subplots()

            sns.lineplot(data=data, x="time", y="p_err_mean", ci="sd", ax=ax)

            ax.set_ylabel(f"p recall error")
            ax.set_ylim(0, 1)

            ax.set_title(teacher_name)

            plt.savefig(fig_path, dpi=300)


if __name__ == "__main__":
    main()
