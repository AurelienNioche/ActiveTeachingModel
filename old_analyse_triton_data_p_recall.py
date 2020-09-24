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


def main():

    force = True

    trial_name = "kiwi"  # input("Trial name: ")
    for subfolder in "Nspec-Nomni", "spec-Nomni":
        for teacher in "recursive_inverse", "threshold":

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

            # teachers = sorted(np.unique(df["Teacher"]))
            # learners = sorted(np.unique(df["Learner"]))
            # psy = sorted(np.unique(df["Psychologist"]))

            # p_recall_error.plot(
            #     teachers=teachers,
            #     learners=learners,
            #     psychologists=psy,
            #     df=df,
            #     fig_path=fig_path)

                # Text psychologist name to column
                # Text learner and teacher combo to row
            fig, ax = plt.subplots()

            sns.lineplot(data=df, x="time", y="p_err_mean", ci="sd")

            # y = [_, df["p_mean_err"].group_by("iter")
            #
            # mean_y = np.mean(y, axis=0)
            # std_y = np.std(y, axis=0)
            #
            # color = "C0"
            #
            # ax.plot(mean_y, color=color)
            #
            # ax.fill_between(
            #     np.arange(len(mean_y)),
            #     mean_y + std_y,
            #     mean_y - std_y,
            #     alpha=0.5,
            #     color=color,
            # )
            #
            # ax.set_ylim((0, 1))
            # ax.set_xlabel("Time")

            ax.set_ylabel(f"p recall error")
            plt.savefig(fig_path)


if __name__ == "__main__":
    main()
