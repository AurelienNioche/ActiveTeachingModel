#%%
import os
from itertools import chain, combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def preprocess_data(data_folder: str, preprocess_data_file: str) -> pd.DataFrame:

    assert os.path.exists(data_folder)

    files = [
        p.path for p in os.scandir(data_folder) if os.path.splitext(p.path)[1] == ".csv"
    ]
    file_count = len(files)

    assert file_count > 0

    row_list = []

    for i, p in tqdm(enumerate(files), total=file_count):
        filename, file_extension = os.path.splitext(p)
        if file_extension != ".csv":
            print(f"ignore {p}")
            continue

        try:
            df = pd.read_csv(p, index_col=[0])

            last_iter = max(df["iter"])
            is_last_iter = df["iter"] == last_iter

            n_learnt = df[is_last_iter]["n_learnt"].iloc[0]

            is_last_ss = df["ss_idx"] == max(df["ss_idx"]) - 1

            ss_n_iter = df["ss_n_iter"][0] - 1

            is_last_iter_ss = df["ss_iter"] == ss_n_iter

            n_learnt_end_ss = df[is_last_ss & is_last_iter_ss]["n_learnt"].iloc[0]

            md_learner = df["md_learner"].iloc[0]
            md_teacher = df["md_teacher"].iloc[0]
            md_psy = df["md_psy"].iloc[0]
            pr_lab = df["pr_lab"].iloc[0]
            pr_val = df["pr_val"].iloc[0]
            pr_lab = eval(pr_lab)
            pr_val = eval(pr_val)
            row = {
                "agent": i,
                "md_learner": md_learner,
                "md_psy": md_psy,
                "md_teacher": md_teacher,
                "n_learnt": n_learnt,
                "n_learnt_end_ss": n_learnt_end_ss,
            }

            for k, v in zip(pr_lab, pr_val):
                row.update({k: v})

            row_list.append(row)

        except Exception as e:
            print(p)
            raise e

    df = pd.DataFrame(row_list)
    df.to_csv(preprocess_data_file)
    print(df)
    return df


def main():
    data_dir = os.path.join("data", "triton", "ukko-run")

    force = False

    dirs = [p for p in os.scandir(data_dir) if p.name != ".DS_Store"]
    dir_names = tuple(dir_.name for dir_ in dirs)
    dirs_combos = tuple(combinations(dir_names, 2))

    pivoted_data_dict = {}
    pivoted_data_mean = pd.Series(index=dir_names)
    for dir_ in dirs:

        print("current dir", dir_.path)

        preprocess_data_file = os.path.join(
            "data", "preprocessed", f"explo_grid_{dir_.name}.csv"
        )
        working_data_dir = dir_.path

        if not os.path.exists(preprocess_data_file) or force:
            df = preprocess_data(
                data_folder=working_data_dir, preprocess_data_file=preprocess_data_file
            )
        else:
            df = pd.read_csv(preprocess_data_file, index_col=[0])

        print("Plotting heatmap...")

        data = pd.DataFrame(
            {"alpha": df["alpha"], "beta": df["beta"], "n_learnt": df["n_learnt"]}
        )
        data = data.round(8).pivot("alpha", "beta", "n_learnt")
        assert len(data.shape) == 2
        dir_name = dir_.name
        pivoted_data_dict[dir_name] = data

    for name in pivoted_data_dict.keys():
        if "leitner" not in name:
            df_plot = (
                pivoted_data_dict[name]
                - pivoted_data_dict["exp_decay-psy_grid-leitner"].values
            )
        else:
            df_plot = pivoted_data_dict["exp_decay-psy_grid-leitner"]
            # df_plot = (
            #     pivoted_data_dict["exp_decay-psy_grid-leitner"]
            #     - pivoted_data_dict["exp_decay-psy_grid-threshold"]
            # )

        plt.close()
        fig, ax = plt.subplots()
        df_plot = df_plot.astype(float)
        mask = df_plot < 0

        sns.heatmap(
            data=df_plot,
            cmap="viridis",
            cbar_kws={"label": "N learnt"},
            ax=ax,
            mask=mask,
        )

        ax.invert_yaxis()

        # ax.set_title(dir_.name)
        ax.set_title(name)
        plt.tight_layout()
        # plt.savefig(os.path.join("fig", f"explo_grid_{dir_.name}.pdf"))
        plt.savefig(os.path.join("fig", f"explo_grid_{name}.pdf"))
    print("Done!")


if __name__ == "__main__":

    main()

    # pivoted_data_mean.loc[dir_name] = data.mean().mean()  # mean in axis=(0, 1)
    # # Subtract the df with lower mean from the other per combination
    # for combo in dirs_combos:

    #     # Select file name with highest mean
    #     assert len(combo) == 2
    #     candidates = pivoted_data_mean.loc[[dir_name for dir_name in combo]]
    #     max_name = candidates.idxmax()
    #     diff_df =
    #
    #     break

    # return
