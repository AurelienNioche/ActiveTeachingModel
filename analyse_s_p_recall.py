import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from plot.subplot import p_recall_error
from settings import paths


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


def main(force=False):

    trial_name = "noprior"  # input("Trial name: ")
    for subfolder in "Nspec-Nomni", "spec-Nomni":
        for teacher in "forward", "threshold":

            teacher_folder = f"exp_decay-psy_grid-{teacher}"
            raw_data_folder = os.path.join("data", "triton", trial_name, subfolder, teacher_folder)
            assert os.path.exists(raw_data_folder)

            preprocessed_folder = os.path.join("data", "preprocessed", "analyse_p_recall")

            preprocess_file = os.path.join(preprocessed_folder, f"{trial_name}_{subfolder}_{teacher}_p_recall.csv")

            fig_folder = os.path.join("fig", "analyse_p_recall")
            fig_path = os.path.join(fig_folder, f"{trial_name}_{subfolder}_{teacher}_p_recall.pdf")

            for f in preprocessed_folder, fig_folder:
                os.makedirs(f, exist_ok=True)

            if not os.path.exists(preprocess_file) or force:
                df = preprocess_data(
                    raw_data_folder=raw_data_folder,
                    preprocess_file=preprocess_file)
            else:
                df = pd.read_csv(preprocess_file, index_col=[0])

            fig, ax = plt.subplots()

            sns.lineplot(data=df, x="time", y="p_err_mean", ci="sd", ax=ax)

            ax.set_ylabel(f"p recall error")
            ax.set_ylim(0, 1)
            plt.savefig(fig_path)


if __name__ == "__main__":
    main()
