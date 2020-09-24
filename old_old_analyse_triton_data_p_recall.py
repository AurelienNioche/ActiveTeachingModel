import os

import numpy as np
import pandas as pd
from tqdm import tqdm

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

        for p_err in df["p_err_mean"]:

            row = {
                "Agent ID": i,
                "Learner": df["md_learner"][0],
                "Psychologist": df["md_psy"][0],
                "Teacher": df["md_teacher"][0],
                "p recall error": p_err,
            }
            row_list.append(row)

    df = pd.DataFrame(row_list)
    df.to_csv(preprocess_file)
    return df


def main():

    force = True

    trial_name = "lin_Nomni_spec"  # input("Trial name: ")
    raw_data_folder = os.path.join(paths.DATA_CLUSTER_DIR, trial_name)

    preprocess_file = os.path.join("data", "preprocessed", f"{trial_name}_p_recall.csv")

    fig_folder = os.path.join("fig", trial_name)
    fig_path = os.path.join(fig_folder, f"{trial_name}_p_recall.pdf")

    os.makedirs(os.path.join("data", "preprocessed"), exist_ok=True)
    os.makedirs(fig_folder, exist_ok=True)

    if not os.path.exists(preprocess_file) or force:
        df = preprocess_data(
            raw_data_folder=raw_data_folder, preprocess_file=preprocess_file
        )
    else:
        df = pd.read_csv(preprocess_file, index_col=[0])

    teachers = sorted(np.unique(df["Teacher"]))
    learners = sorted(np.unique(df["Learner"]))
    psy = sorted(np.unique(df["Psychologist"]))

    p_recall_error.plot(
        teachers=teachers,
        learners=learners,
        psychologists=psy,
        df=df,
        fig_path=fig_path,
    )


if __name__ == "__main__":
    main()
