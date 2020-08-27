import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from plot.subplot import chocolate, box, efficiency_cost, p_recall_error, hist


# EXT = '_exp'
# EXT = "_exp_omni_spec"
DATA_NAME = "exp_Nomni_spec"

FORCE = False

RAW_DATA_FOLDER = os.path.join("data", "triton", DATA_NAME)
PREPROCESSED_DATA_FILE = os.path.join(
    "data", "preprocessed", f"chocolate_triton{DATA_NAME}.csv"
)
FIG_CHOCOLATE_PATH = os.path.join("fig", f"{DATA_NAME}_chocolate.pdf")
FIG_BOXPLOT_PATH = os.path.join("fig", f"{DATA_NAME}_boxplot.pdf")

FIG_CHOCOLATE_PATH_END_PR_SS = os.path.join("fig", f"{DATA_NAME}_chocolate_end_pr_ss.pdf")
FIG_BOXPLOT_PATH_END_PR_SS = os.path.join("fig", f"{DATA_NAME}_boxplot_end_pr_ss.pdf")

<<<<<<< HEAD
FIG_CHOCOLATE_PATH_END_PR_SS = os.path.join(
    "fig", f"chocolate_triton{EXT}_end_pr_ss.pdf"
)
FIG_BOXPLOT_PATH_END_PR_SS = os.path.join("fig", f"boxplot_triton{EXT}_end_pr_ss.pdf")
=======
FIG_HIST_PATH = os.path.join("fig", f"{DATA_NAME}_hist.pdf")
>>>>>>> 1ba4015d5f43ee30c0056584cef157134e2769e4

os.makedirs(os.path.join("data", "preprocessed"), exist_ok=True)


def preprocess_data():

    assert os.path.exists(RAW_DATA_FOLDER)

    path, dirs, files = next(os.walk(RAW_DATA_FOLDER))
    file_count = len(files)

    assert file_count > 0

    row_list = []

    for i, p in tqdm(enumerate(os.scandir(RAW_DATA_FOLDER)), total=file_count):
        df = pd.read_csv(p.path, index_col=[0])

        # max_iter = max(df["iter"])
        # last_iter = df[df["iter"] == max_iter]
        # n_learnt = last_iter["n_learnt"].iloc[0]

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
            "Items learnt": n_learnt,
            "Items learnt end prev ss": n_learnt_end_ss,
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

    items_learnt = "Items learnt"
    items_learnt_last_session = "Items learnt end prev ss"

<<<<<<< HEAD
    chocolate.plot(
        df=df,
        teachers=teachers,
        learners=learners,
        psychologists=psy,
        fig_path=FIG_CHOCOLATE_PATH,
        learnt_label=items_learnt,
    )

    box.plot(df=df, fig_path=FIG_BOXPLOT_PATH, learnt_label=items_learnt)

    chocolate.plot(
        df=df,
        teachers=teachers,
        learners=learners,
        psychologists=psy,
        fig_path=FIG_CHOCOLATE_PATH_END_PR_SS,
        learnt_label=items_learnt_last_session,
    )

    box.plot(
        df=df,
        fig_path=FIG_BOXPLOT_PATH_END_PR_SS,
        learnt_label=items_learnt_last_session,
    )
=======
    hist.plot(learnt_label=items_learnt, df=df,
              fig_path=FIG_HIST_PATH)

    # chocolate.plot(
    #     df=df,
    #     teachers=teachers,
    #     learners=learners,
    #     psychologists=psy,
    #     fig_path=FIG_CHOCOLATE_PATH,
    #     learnt_label=items_learnt
    # )
    #
    # box.plot(df=df, fig_path=FIG_BOXPLOT_PATH, learnt_label=items_learnt)
    #
    # chocolate.plot(
    #     df=df,
    #     teachers=teachers,
    #     learners=learners,
    #     psychologists=psy,
    #     fig_path=FIG_CHOCOLATE_PATH_END_PR_SS,
    #     learnt_label=items_learnt_last_session
    # )
    #
    # box.plot(df=df, fig_path=FIG_BOXPLOT_PATH_END_PR_SS,
    #          learnt_label=items_learnt_last_session)
>>>>>>> 1ba4015d5f43ee30c0056584cef157134e2769e4


if __name__ == "__main__":
    main()
