import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from plot.subplot import chocolate, box, hist

DATA_NAME = "exp_no_beta_always_recall"

FORCE = False

RAW_DATA_FOLDER = os.path.join("data", "triton", DATA_NAME)
PREPROCESSED_DATA_FILE = os.path.join(
    "data", "preprocessed", f"chocolate_triton{DATA_NAME}.csv")

os.makedirs(os.path.join("data", "preprocessed"), exist_ok=True)


def preprocess_data():

    assert os.path.exists(RAW_DATA_FOLDER)

    path, dirs, files = next(os.walk(RAW_DATA_FOLDER))
    file_count = len(files)

    assert file_count > 0

    row_list = []

    for i, p in tqdm(enumerate(os.scandir(RAW_DATA_FOLDER)), total=file_count):
        df = pd.read_csv(p.path, index_col=[0])

        last_ss_idx = max(df["ss_idx"])
        is_last_ss = df["ss_idx"] == last_ss_idx
        is_first_iter_ss = df["ss_iter"] == 0

        n_learnt = df[is_last_ss & is_first_iter_ss]["n_learnt"].iloc[0]

        is_before_last_ss = df["ss_idx"] == last_ss_idx - 1
        is_last_iter_ss = df["ss_iter"] == df["ss_n_iter"][0] - 1

        n_learnt_end_ss = df[is_before_last_ss & is_last_iter_ss]["n_learnt"].iloc[0]

        row = {
            "Agent ID": i,
            "Learner": df["md_learner"][0],
            "Psychologist": df["md_psy"][0],
            "Teacher": df["md_teacher"][0],
            "Items learnt one day later": n_learnt,
            "Items learnt end last session": n_learnt_end_ss,
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

    items_learnt = \
        {
            "one_day_later": "Items learnt one day later",
            "end_last_ss": "Items learnt end last session"
        }

    for k, v in items_learnt.items():

        fig_path = os.path.join("fig",
                                f"{DATA_NAME}_chocolate_{k}.pdf")
        chocolate.plot(
            df=df,
            teachers=teachers,
            learners=learners,
            psychologists=psy,
            fig_path=fig_path,
            learnt_label=v,
        )

        fig_path = os.path.join("fig",
                                f"{DATA_NAME}_box_{k}.pdf")

        box.plot(
            df=df,
            fig_path=fig_path,
            learnt_label=v,
        )

        fig_path = os.path.join("fig",
                                f"{DATA_NAME}_hist_{k}.pdf")

        hist.plot(learnt_label=v, df=df,
                  fig_path=fig_path)


if __name__ == "__main__":
    main()
