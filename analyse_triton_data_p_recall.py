import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from plot.subplot import p_recall_error

DATA_NAME = "exp_Nomni_spec"

FORCE = False

RAW_DATA_FOLDER = os.path.join("data", "triton", DATA_NAME)

PREPROCESSED_DATA_FILE = os.path.join(
    "data", "preprocessed", f"{DATA_NAME}_p_recall.csv")

FIG_PATH = os.path.join("fig", f"{DATA_NAME}_p_recall.pdf")

os.makedirs(os.path.join("data", "preprocessed"), exist_ok=True)


def preprocess_data():

    assert os.path.exists(RAW_DATA_FOLDER)

    path, dirs, files = next(os.walk(RAW_DATA_FOLDER))
    file_count = len(files)

    assert file_count > 0

    row_list = []

    for i, p in tqdm(enumerate(os.scandir(RAW_DATA_FOLDER)), total=file_count):
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
    df.to_csv(PREPROCESSED_DATA_FILE)
    return df


def main():

    if not os.path.exists(PREPROCESSED_DATA_FILE) or FORCE:
        df = preprocess_data()
    else:
        df = pd.read_csv(PREPROCESSED_DATA_FILE, index_col=[0])

    teachers = sorted(np.unique(df["Teacher"]))
    learners = sorted(np.unique(df["Learner"]))
    psy = sorted(np.unique(df["Psychologist"]))

    p_recall_error.plot(
        teachers=teachers,
        learners=learners,
        psychologists=psy,
        df=df,
        fig_path=FIG_PATH)


if __name__ == "__main__":
    main()
