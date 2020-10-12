import os

import numpy as np
import pandas as pd
from tqdm import tqdm


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
                "p_err_mean": p_err,
                "time": t,
            }
            row_list.append(row)

    df = pd.DataFrame(row_list)
    df.to_csv(preprocess_file)
    return df


def get_data(dataset_name, condition_name, teacher_name, force=False):
    teacher_folder = f"exp_decay-psy_grid-{teacher_name}"
    raw_data_folder = os.path.join("data", "triton",
                                   dataset_name, condition_name,
                                   teacher_folder)
    assert os.path.exists(raw_data_folder)

    preprocessed_folder = os.path.join("data", "preprocessed",
                                       "analyse_p_recall")

    preprocess_file = os.path.join(
        preprocessed_folder,
        f"{dataset_name}_{condition_name}_{teacher_name}_p_recall.csv")

    os.makedirs(preprocessed_folder, exist_ok=True)

    if not os.path.exists(preprocess_file) or force:
        df = preprocess_data(
            raw_data_folder=raw_data_folder,
            preprocess_file=preprocess_file)
    else:
        df = pd.read_csv(preprocess_file, index_col=[0])

    return df
