#%%
"""
Make boxplot for when having
separate teacher directories.
"""

import os

import pandas as pd
import seaborn as sns
from tqdm import tqdm

import settings.paths as paths
from plot.subplot import box

preprocessed_dir = os.path.join(paths.DATA_DIR, "preprocessed")
preprocessed_f_names = os.listdir(os.path.join(paths.DATA_DIR, "preprocessed"))

# for f_name in preprocessed_f_names:
#         f_path = os.path.join(preprocessed_dir, f_name)
#         df = pd.read_csv(f_path, index_col=[0])
#
#         fig_path = os.path.join(f"{f_name.}")
#         box.plot(df=df, fig_path=fig_path, learnt_label=v)
#         box.plot(df=df, fig_path=fig_path, learnt_label=description)
#
# items_learnt = {
#         "one_day_later": "Items learnt one day later",
#         "end_last_ss": "Items learnt end last session",
# }

for trial_dir_name in tqdm(os.listdir(paths.DATA_CLUSTER_DIR)):
    print(f"Loading df from {trial_dir_name}")
    trial_path = os.path.join(paths.DATA_CLUSTER_DIR, trial_dir_name)

    trial_dfs = []
    for cond_dir_name in os.listdir(trial_path):
        print(f"... {cond_dir_name}")
        cond_path = os.path.join(trial_path, cond_dir_name)

        # Take file paths for reduce
        csv_dfs = []
        for csv_name in os.listdir(cond_path):
            csv_path = os.path.join(cond_path, csv_name)
            csv_dfs.append(pd.read_csv(csv_path, index_col=0))

        # Reduce each run df into condition df
        cond_df = csv_dfs[0]
        for idx, csv_df in enumerate(csv_dfs):
            if idx == 0:
                continue
            cond_df = cond_df.append(csv_df)

        trial_dfs.append(cond_df)

    # Reduce condition df into trial df
    trial_df = trial_dfs[0]
    for idx, cond_df in enumerate(trial_dfs):
        if idx == 0:
            continue
        trial_df.append(cond_df)

    # Preprocessing
    last_iter = max(trial_df["iter"])
    is_last_iter = trial_df["iter"] == last_iter

    n_learnt = trial_df[is_last_iter]["n_learnt"].iloc[0]

    is_last_ss = trial_df["ss_idx"] == max(trial_df["ss_idx"]) - 1

    ss_n_iter = trial_df["ss_n_iter"][0] - 1

    is_last_iter_ss = trial_df["ss_iter"] == ss_n_iter

    n_learnt_end_ss = trial_df[is_last_ss & is_last_iter_ss]["n_learnt"].iloc[0]

    row = {
        "Agent ID": i,
        "Learner": trial_df["md_learner"][0],
        "Psychologist": trial_df["md_psy"][0],
        "Teacher": trial_df["md_teacher"][0],
        "Items learnt one day later": n_learnt,
        "Items learnt end last session": n_learnt_end_ss,
    }
    print(trial_df)


#     for t_cond, description in items_learnt.items():
#         fig_path = os.path.join(f"{trial_dir_name}")
#         plot(df=trial_df, fig_path=fig_path, learnt_label=description)
