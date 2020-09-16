"""
User parameter recovery
"""

import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import settings.paths as paths


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
    hist: Iterable,
    success: Iterable,
    timestamp: Iterable,
    cst_time: float,
    eps: float,
):
    """Compute log-likelihood for one param bounds pair"""

    a, b = param

    log_p_hist = np.zeros(len(hist))

    for item in np.unique(hist):

        is_item = hist == item
        rep = timestamp[is_item]
        n = len(rep)

        log_p_item = np.zeros(n)
        # !!! To adapt for xp
        log_p_item[0] = -np.inf  # whatever the model, p=0
        # !!! To adapt for xp
        for i in range(1, n):
            delta_rep = rep.iloc[i] - rep.iloc[i - 1]
            fr = a * (1 - b) ** (i - 1)
            delta_rep *= cst_time
            log_p_item[i] = -fr * delta_rep

        log_p_hist[is_item] = log_p_item
    p_hist = np.exp(log_p_hist)
    failure = np.invert(success.astype(bool))
    p_hist[failure.values] = 1 - p_hist[failure.values]
    _log_lik = np.log(p_hist + eps)
    sum_ll = _log_lik.sum()
    return sum_ll


def get_all_log_lik(user_df: pd.DataFrame, grid_df: pd.DataFrame, eps: float) -> list:
    """Compute log-likelihood for all grid values"""

    print("Computing log-likelihood for all param pairs...")
    sums_ll = []
    beginning_history = pd.Timestamp("1970-01-01", tz="UTC")
    # 5s is good
    one_s = pd.Timedelta("1s")
    for _, param_pair in tqdm(grid_df.iterrows()):
        sums_ll.append(
            log_lik(
                param_pair,
                user_df["item"],
                user_df["success"],
                (user_df["ts_reply"] - beginning_history)
                // one_s,  # To seconds as in pandas docs
                1,  # 1 / (5 * 60 ** 2),
                eps,
            )
        )
    print("Done!")

    return sums_ll


def plot_param_space(user_name: str, grid: pd.DataFrame, log_liks: np.ndarray) -> None:
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
    plt.savefig(os.path.join("fig", f"param_grid_{user_name}.pdf"))
    print("Done!")


def main(f_results: str) -> (pd.DataFrame, pd.DataFrame):
    """Get grid values, log-likelihood, and plot heatmap"""

    eps = np.finfo(np.float).eps

    # Grid
    bounds = np.array([[0.0000001, 100.0], [0.0001, 0.99]])
    grid_size = 20
    methods = np.array([np.geomspace, np.linspace])  # Use log scale for alpha
    grid = cp_grid_param_loglin(grid_size, bounds, methods)
    grid = pd.DataFrame(grid, columns=("alpha", "beta"))

    # Log-likelihood
    results_df = pd.read_csv(os.path.join(paths.DATA_DIR, f_results), index_col=[0])
    results_df["ts_display"] = pd.to_datetime(
        results_df["ts_display"]
    )  # str to datetime
    results_df["ts_reply"] = pd.to_datetime(results_df["ts_reply"])  # str to datetime
    likelihoods = {
        user: get_all_log_lik(user_df, grid, eps)
        for user, user_df in results_df.groupby("user")
        if "carlos2" not in user
    }

    # Plot
    users = list(likelihoods.keys())
    for user in users:
        plot_param_space(user, grid, likelihoods[user])

    return likelihoods, grid


if __name__ == "__main__":
    loglikes, grid_ = main("user/data_full.csv")


# def enclose_log_lik_df(df, eps):
#     def log_lik_closure(param_pair):
#         return log_lik(param_pair, df["item"], df["success"], df["ts_reply"], np.ones_like(df["item"]), eps)
#     return log_lik_closure
# log_lik_cl = enclose_log_lik_df(df_, EPS)
#
# def cp_grid_param(grid_size, bounds):
#     diff = bounds[:, 1] - bounds[:, 0] > 0
#     not_diff = np.invert(diff)
#
#     values = np.atleast_2d([np.linspace(*b, num=grid_size) for b in bounds[diff]])
#     var = cartesian_product(*values)
#     grid = np.zeros((max(1, len(var)), len(bounds)))
#     if np.sum(diff):
#         grid[:, diff] = var
#     if np.sum(not_diff):
#         grid[:, not_diff] = bounds[not_diff, 0]
#
#     return grid
#%%
# plt.close()
# user_name = "carlos@test.com"
#
# beginning_history = pd.Timestamp("1970-01-01", tz="UTC")
# one_s = pd.Timedelta("1s")
#
# results_df = pd.read_csv(os.path.join(paths.DATA_DIR, "results.csv"), index_col=[0])
# results_df = results_df.query("user == @user_name")
# results_df["ts_reply"] = pd.to_datetime(results_df["ts_reply"])
# results_df["ts_reply"] = (results_df["ts_reply"] - beginning_history) // one_s
# sessions_arrs = []
# for n_session, sub_df in results_df.groupby("session"):
#     sessions_arrs.append(sub_df["ts_reply"].diff().values)
# # sessions_arr = np.hstack(sessions_arrs)
# sessions_arr = pd.Series(np.hstack(sessions_arrs), name="ts_reply")
# sns.distplot(sessions_arr[sessions_arr < 50], bins=100, kde=False)
# # plt.hist(sessions_arr[sessions_arr < 50], bins=30)
# plt.savefig(os.path.join("fig", f"histo-reply-{user_name}.pdf"))
# print(sessions_arr.value_counts())
