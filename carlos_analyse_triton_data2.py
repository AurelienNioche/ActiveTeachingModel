import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.preprocess import preprocess_data
from plot.subplot import box, chocolate, hist
from settings import paths

FORCE = True


def main(trial_name, cond_name):

    # cond_name = input("Trial name: ")

    # raw_data_folder = os.path.join(paths.DATA_CLUSTER_DIR, cond_name)

    for trial_dir_name in tqdm(os.listdir(paths.DATA_CLUSTER_DIR)):
        print(f"Loading df from {trial_dir_name}")
        trial_path = os.path.join(paths.DATA_CLUSTER_DIR, trial_dir_name)

        for cond_dir_name in os.listdir(trial_path):
            print(f"... {cond_dir_name}")
            cond_path = os.path.join(trial_path, cond_dir_name)

            raw_data_folder = os.path.join(
                paths.DATA_CLUSTER_DIR, trial_dir_name, cond_dir_name
            )
            fig_folder = os.path.join("fig", cond_name)

            preprocess_data_file = os.path.join(
                "data", "preprocessed", f"triton_{cond_name}.csv"
            )

            os.makedirs(os.path.join("data", "preprocessed"), exist_ok=True)
            os.makedirs(os.path.join("fig", cond_name), exist_ok=True)

            if not os.path.exists(preprocess_data_file) or FORCE:
                print(preprocess_data_file)
                df = preprocess_data(
                    raw_data_folder=raw_data_folder,
                    preprocess_data_file=preprocess_data_file,
                )
            else:
                df = pd.read_csv(preprocess_data_file, index_col=[0])

            teachers = sorted(np.unique(df["Teacher"]))
            learners = sorted(np.unique(df["Learner"]))
            psy = sorted(np.unique(df["Psychologist"]))

            items_learnt = {
                "one_day_later": "Items learnt one day later",
                "end_last_ss": "Items learnt end last session",
            }

            for k, v in items_learnt.items():

                # fig_path = os.path.join(fig_folder, f"{cond_name}_chocolate_{k}.pdf")
                # chocolate.plot(
                #     df=df,
                #     teachers=teachers,
                #     learners=learners,
                #     psychologists=psy,
                #     fig_path=fig_path,
                #     learnt_label=v,
                # )

                plt.close()

                fig_path = os.path.join(fig_folder, f"{cond_name}_box_{k}.pdf")
                box.plot(df=df, fig_path=fig_path, learnt_label=v)

                fig_path = os.path.join(fig_folder, f"{cond_name}_hist_{k}.pdf")
                hist.plot(learnt_label=v, df=df, fig_path=fig_path)


if __name__ == "__main__":
    for trial_dir_name in paths.DATA_CLUSTER_DIR:
        for cond_dir_name in os.listdir(f"./data/triton/{trial_dir_name}"):
            main(trial_dir_name, cond_dir_name)
