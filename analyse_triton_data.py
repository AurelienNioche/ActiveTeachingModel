import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from plot.subplot import chocolate, box, hist
from settings import paths

FORCE = False


def preprocess_data(preprocess_data_file, raw_data_folder):

    assert os.path.exists(raw_data_folder)

    path, dirs, files = next(os.walk(raw_data_folder))
    file_count = len(files)

    assert file_count > 0

    row_list = []

    for i, p in tqdm(enumerate(os.scandir(raw_data_folder)), total=file_count):
        df = pd.read_csv(p.path, index_col=[0])

        last_iter = max(df["iter"])
        is_last_iter = df["iter" == last_iter]

        n_learnt = df[is_last_iter]["n_learnt"].iloc[0]

        is_last_ss = df["ss_idx"] == max(df["ss_idx"]) - 1
        is_last_iter_ss = df["ss_iter"] == df["ss_n_iter"][0] - 1

        n_learnt_end_ss = df[is_last_ss & is_last_iter_ss]["n_learnt"].iloc[0]

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
    df.to_csv(preprocess_data_file)
    return df


def main():

    trial_name = input("Trial name: ")
    raw_data_folder = os.path.join(paths.DATA_CLUSTER_DIR, trial_name)

    # RAW_DATA_FOLDER = "/Volumes/niochea1/ActiveTeachingModel/data/triton"
    # #os.path.join("data", "triton")
    preprocess_data_file = os.path.join(
        "data", "preprocessed", f"chocolate_triton{trial_name}.csv")

    os.makedirs(os.path.join("data", "preprocessed"), exist_ok=True)

    if not os.path.exists(preprocess_data_file) or FORCE:
        df = preprocess_data(
            raw_data_folder=raw_data_folder,
            preprocess_data_file=preprocess_data_file)
    else:
        df = pd.read_csv(preprocess_data_file, index_col=[0])

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
                                f"{trial_name}_chocolate_{k}.pdf")
        chocolate.plot(
            df=df,
            teachers=teachers,
            learners=learners,
            psychologists=psy,
            fig_path=fig_path,
            learnt_label=v,
        )

        fig_path = os.path.join("fig",
                                f"{trial_name}_box_{k}.pdf")

        box.plot(
            df=df,
            fig_path=fig_path,
            learnt_label=v,
        )

        fig_path = os.path.join("fig",
                                f"{trial_name}_hist_{k}.pdf")

        hist.plot(learnt_label=v, df=df,
                  fig_path=fig_path)


if __name__ == "__main__":
    main()
