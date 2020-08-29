import os
import sys

import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

FORCE = True

SCRIPT_NAME = os.path.basename(__file__).split(".")[0]

FIG_FOLDER = os.path.join("fig", SCRIPT_NAME)
os.makedirs(FIG_FOLDER, exist_ok=True)


def main():

    raw_data_folder = os.path.join("data", "local_walsh")
    path, dirs, files = next(os.walk(raw_data_folder))

    labels = []
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(raw_data_folder, f), index_col=[0])
        dfs.append(df)
        label = os.path.splitext(f)[0]
        labels.append(label)

    fig, ax = plt.subplots()
    for df, label in zip(dfs, labels):
        ax.plot(df['timestamp'], df['n_learnt'], label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("n learnt")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()