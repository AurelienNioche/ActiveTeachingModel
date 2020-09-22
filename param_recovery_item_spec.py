"""
User parameter recovery
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]


FIG_FOLDER = os.path.join("fig", SCRIPT_NAME)
os.makedirs(FIG_FOLDER, exist_ok=True)

BKP_FOLDER = os.path.join("bkp", SCRIPT_NAME)
os.makedirs(BKP_FOLDER, exist_ok=True)


EPS = np.finfo(np.float).eps


def cartesian_product(*arrays):

    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def cp_grid_param_loglin(grid_size, bounds, methods):
    """Get grid parameters, with log scale for alpha"""

    diff = bounds[:, 1] - bounds[:, 0] > 0
    not_diff = np.invert(diff)

    values = np.atleast_2d(
        [m(*b, num=grid_size) for (b, m) in zip(bounds[diff], methods[diff])]
    )
    var = cartesian_product(*values)
    grid = np.zeros((max(1, len(var)), len(bounds)))
    if np.sum(diff):
        grid[:, diff] = var
    if np.sum(not_diff):
        grid[:, not_diff] = bounds[not_diff, 0]

    return grid


def get_all_log_lik(results_df: pd.DataFrame,
                    grid_df: pd.DataFrame) -> pd.DataFrame:
    """Compute log-likelihood for all grid values"""

    row_list = []

    for user, user_df in results_df.groupby("user"):

        if '@test' in user or user_df.n_session_done.iloc[0] != 14:
            continue

        print("user", user)

        sums_ll = []

        # Convert timestamps into seconds
        beginning_history = pd.Timestamp("1970-01-01", tz="UTC")
        ts = (user_df["ts_reply"] - beginning_history).dt.total_seconds().values

        hist = user_df["item"].values

        success = user_df["success"].values.astype(bool)

        for idx_pair, param_pair in tqdm(grid_df.iterrows(), file=sys.stdout):

            a, b = param_pair

            # For each item
            for item in np.unique(hist):

                is_item = hist == item
                rep = ts[is_item]
                n = len(rep)

                log_p_item = np.zeros(n)
                # !!! To adapt for xp
                log_p_item[0] = 0
                # -np.inf  # whatever the model, p=0
                # !!! To adapt for xp
                for i in range(1, n):
                    delta_rep = rep[i] - rep[i - 1]
                    fr = a * (1 - b) ** (i - 1)
                    # delta_rep *= cst_time
                    log_p_item[i] = -fr * delta_rep

                p_item = np.exp(log_p_item)
                failure = np.invert(success[is_item])
                p_item[failure] = 1 - p_item[failure]
                _log_lik = np.log(p_item + EPS)
                lls = _log_lik.sum()

                row_list.append({
                    "alpha": a,
                    "beta": b,
                    "item": item,
                    "idx_pair": idx_pair,
                    "lls": lls,
                    "user": user})

    return pd.DataFrame(row_list)


def plot_param_space(user_name: str, grid: pd.DataFrame,
                     log_liks: np.ndarray) -> None:
    """Heatmap of the alpha-beta parameter space"""

    print("Plotting heatmap...")
    plt.close()
    log_liks = pd.Series(log_liks, name="log_lik")
    data = pd.concat((grid, log_liks), axis=1)
    try:  # Duplicated entries can appear with rounding
        data = data.round(2).pivot("alpha", "beta", "log_lik")
    except:
        data = data.pivot("alpha", "beta", "log_lik")
    ax = sns.heatmap(data=data, cmap="viridis")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_FOLDER,
                             f"{user_name}.pdf"))
    print("Done!")


def analyse(f_results):

    # Grid
    bounds = np.array([[0.0000001, 0.025], [0.0001, 0.9999]])
    grid_size = 20
    methods = np.array([np.geomspace, np.linspace])  # Use log scale for alpha
    grid = cp_grid_param_loglin(grid_size, bounds, methods)
    grid_df = pd.DataFrame(grid, columns=("alpha", "beta"))

    # Log-likelihood
    results_df = pd.read_csv(f_results, index_col=[0])
    results_df["ts_display"] = pd.to_datetime(
        results_df["ts_display"]
    )  # str to datetime
    results_df["ts_reply"] = pd.to_datetime(results_df["ts_reply"])  # str to datetime

    lls_df = get_all_log_lik(results_df=results_df,
                             grid_df=grid_df)

    lls_df.to_csv(os.path.join(BKP_FOLDER, "lls_item_spec.csv"))
    grid_df.to_csv(os.path.join(BKP_FOLDER, "grid.csv"))

    return lls_df, grid_df


def main(f_results: str) -> (pd.DataFrame, pd.DataFrame):
    """Get grid values, log-likelihood, and plot heatmap"""

    if not os.path.exists("lls_item_spec.csv"):
        lls_df, grid_df = analyse(f_results=f_results)

    else:
        lls_df = pd.read_csv(os.path.join(BKP_FOLDER, "lls_item_spec.csv"))
        grid_df = pd.read_csv(os.path.join(BKP_FOLDER, "grid.csv"))

    row_list = []
    for (user, item), df in lls_df.groupby(["user", "item"]):
        best_idx = np.argmax(df["lls"].values)
        alpha = df["alpha"].values[best_idx]
        beta = df["beta"].values[best_idx]
        row_list.append({
            "user": user,
            "item": item,
            "alpha": alpha,
            "beta": beta
        })

    bp_item_spec_df = pd.DataFrame(row_list)
    bp_item_spec_df.to_csv(os.path.join(BKP_FOLDER, "bp_item_spec.csv"))

    row_list = []

    for user, df in lls_df.groupby("user"):
        for (alpha, beta), df_pair in df.groupby(['alpha', 'beta']):
            best_idx = np.argmax(df_pair['lls'].sum().values)
            best_alpha_user.append(gp['alpha'].unique()[best_idx])
            best_beta_user.append(gp["beta"].unique()[best_idx])

    fig, ax = plt.subplots()

    sns.scatterplot(data=bp_item_spec_df,
                    x="beta",
                    y="alpha", alpha=0.01, color="C0", ax=ax)
    # ax.scatter(best_beta_user, best_alpha_user, color="red")
    #
    ax.set_xlabel("beta")
    ax.set_yscale('log')
    ax.set_ylabel("alpha")
    ax.set_xlim(min(grid_df["beta"]), max(grid_df["beta"]))
    ax.set_ylim(min(grid_df["alpha"]), max(grid_df["alpha"]))

    plt.savefig(os.path.join(FIG_FOLDER, "scatter.png"), dpi=300)


if __name__ == "__main__":
    main("data_full.csv")

