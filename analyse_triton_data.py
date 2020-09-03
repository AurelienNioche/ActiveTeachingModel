import os

import numpy as np
import pandas as pd

from analysis.preprocess import preprocess_data
from plot.subplot import box, chocolate, hist
from settings import paths

FORCE = True


def main():

    trial_name = "lin_Nomni_Nspec"   # input("Trial name: ")
    raw_data_folder = os.path.join(paths.DATA_CLUSTER_DIR, trial_name)
    fig_folder = os.path.join("fig", trial_name)

    preprocess_data_file = os.path.join(
        "data", "preprocessed", f"chocolate_triton_{trial_name}.csv")

    os.makedirs(os.path.join("data", "preprocessed"), exist_ok=True)
    os.makedirs(os.path.join("fig", trial_name), exist_ok=True)

    if not os.path.exists(preprocess_data_file) or FORCE:
        print(preprocess_data_file)
        df = preprocess_data(
            raw_data_folder=raw_data_folder, preprocess_data_file=preprocess_data_file
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

        # fig_path = os.path.join(fig_folder, f"{trial_name}_chocolate_{k}.pdf")
        # chocolate.plot(
        #     df=df,
        #     teachers=teachers,
        #     learners=learners,
        #     psychologists=psy,
        #     fig_path=fig_path,
        #     learnt_label=v,
        # )

        fig_path = os.path.join(fig_folder, f"{trial_name}_box_{k}.pdf")
        box.plot(df=df, fig_path=fig_path, learnt_label=v)

        fig_path = os.path.join(fig_folder, f"{trial_name}_hist_{k}.pdf")
        hist.plot(learnt_label=v, df=df, fig_path=fig_path)


if __name__ == "__main__":
    main()
