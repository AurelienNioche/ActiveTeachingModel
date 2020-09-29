import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm

FORCE = True

SCRIPT_NAME = os.path.basename(__file__).split(".")[0]

FIG_FOLDER = os.path.join("fig", SCRIPT_NAME)
os.makedirs(FIG_FOLDER, exist_ok=True)


def main():

    raw_data_folder = os.path.join("data", "triton", "walsh_new")
    path, dirs, files = next(os.walk(raw_data_folder))

    print("n files", len(files))

    teachers = []
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(raw_data_folder, f), index_col=[0])
        dfs.append(df)
        fn = os.path.splitext(f)[0]
        _, date, learner, psy, teacher, agent_id = fn.split("-")
        teachers.append(teacher)

    colors = {t: f"C{i}" for (i, t) in enumerate(np.unique(teachers))}

    fig, ax = plt.subplots()
    for df, teacher in zip(dfs, teachers):
        ax.plot(
            df["timestamp"],
            df["n_learnt"],
            label=teacher,
            color=colors[teacher],
            alpha=0.1,
            lw=0.5,
        )
    ax.set_xlabel("time")
    ax.set_ylabel("n learnt")

    legend_elements = [
        Line2D([0], [0], color=v, lw=1, label=k) for (k, v) in colors.items()
    ]
    # Line2D([0], [0], marker='o', color='w', label='Scatter',
    #        markerfacecolor='g', markersize=15),
    # Patch(facecolor='orange', edgecolor='r',
    #       label='Color Patch')

    ax.legend(handles=legend_elements)

    plt.show()


if __name__ == "__main__":
    main()