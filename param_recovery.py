"""
User parameter recovery
"""

import os
import sys

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


FIG_FOLDER = os.path.join("fig", "param_recovery")
os.makedirs(FIG_FOLDER, exist_ok=True)


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


def log_lik(
    param: Iterable,
    hist: np.ndarray,
    success: np.ndarray,
    timestamp: np.ndarray,
    cst_time=1.0,
    eps=np.finfo(np.float).eps,
):
    """Compute log-likelihood for one param bounds pair"""

    a, b = param

    log_p_hist = np.zeros(len(hist))

    # For each item
    for item in np.unique(hist):

        is_item = hist == item
        rep = timestamp[is_item]
        n = len(rep)

        log_p_item = np.zeros(n)
        # !!! To adapt for xp
        log_p_item[0] = 0
        # -np.inf  # whatever the model, p=0
        # !!! To adapt for xp
        for i in range(1, n):
            delta_rep = rep[i] - rep[i - 1]
            fr = a * (1 - b) ** (i - 1)
            delta_rep *= cst_time
            log_p_item[i] = -fr * delta_rep

        log_p_hist[is_item] = log_p_item
    p_hist = np.exp(log_p_hist)
    failure = np.invert(success)
    p_hist[failure] = 1 - p_hist[failure]
    _log_lik = np.log(p_hist + eps)
    sum_ll = _log_lik.sum()
    return sum_ll


def get_all_log_lik(user_df: pd.DataFrame,
                    grid_df: pd.DataFrame) -> list:
    """Compute log-likelihood for all grid values"""

    sums_ll = []

    # Convert timestamps into seconds
    beginning_history = pd.Timestamp("1970-01-01", tz="UTC")
    ts = (user_df["ts_reply"] - beginning_history).dt.total_seconds().values

    for _, param_pair in tqdm(grid_df.iterrows(), file=sys.stdout):
        sums_ll.append(
            log_lik(
                param_pair,
                user_df["item"].values,
                user_df["success"].values.astype(bool),
                ts))

    return sums_ll


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


def main(f_results: str) -> (pd.DataFrame, pd.DataFrame):
    """Get grid values, log-likelihood, and plot heatmap"""

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

    likelihoods = {}

    for user, user_df in results_df.groupby("user"):

        if '@test' not in user and user_df.n_session_done.iloc[0] == 14:
            print("user", user)

            lls = get_all_log_lik(user_df, grid_df)
            likelihoods[user] = lls

    ll_df = pd.DataFrame(likelihoods)
    ll_df.to_csv("lls.csv")

    grid_df.to_csv("grid.csv")

    # Plot
    users = list(likelihoods.keys())
    for user in users:
        plot_param_space(user, grid_df, likelihoods[user])

    return likelihoods, grid_df


if __name__ == "__main__":
    main("data/human/data_full.csv")

