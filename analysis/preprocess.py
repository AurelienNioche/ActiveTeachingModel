import os

import pandas as pd
from tqdm import tqdm


def preprocess_data(preprocess_data_file, raw_data_folder):

    assert os.path.exists(raw_data_folder)

    path, dirs, files = next(os.walk(raw_data_folder))
    file_count = len(files)

    assert file_count > 0

    row_list = []

    for i, p in tqdm(enumerate(os.scandir(raw_data_folder)), total=file_count):
        filename, file_extension = os.path.splitext(p)
        if file_extension != '.csv':
            print(f"ignore {p}")
            continue

        try:
            df = pd.read_csv(p.path, index_col=[0])

            last_iter = max(df["iter"])
            is_last_iter = df["iter"] == last_iter

            n_learnt = df[is_last_iter]["n_learnt"].iloc[0]

            is_last_ss = df["ss_idx"] == max(df["ss_idx"]) - 1

            ss_n_iter = df["ss_n_iter"][0] - 1

            is_last_iter_ss = df["ss_iter"] == ss_n_iter

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

        except Exception as e:
            print(p)
            raise e

    df = pd.DataFrame(row_list)
    df.to_csv(preprocess_data_file)
    return df
