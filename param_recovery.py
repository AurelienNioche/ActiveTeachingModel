#%%

import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import settings.paths as paths


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

def cp_grid_param_loglin(grid_size, bounds, methods):
    diff = bounds[:, 1] - bounds[:, 0] > 0
    not_diff = np.invert(diff)

    values = np.atleast_2d([m(*b, num=grid_size) for (b, m) in zip(bounds[diff], methods[diff])])
    var = cartesian_product(*values)
    grid = np.zeros((max(1, len(var)), len(bounds)))
    if np.sum(diff):
        grid[:, diff] = var
    if np.sum(not_diff):
        grid[:, not_diff] = bounds[not_diff, 0]

    return grid

def cp_grid_param(grid_size, bounds):
    diff = bounds[:, 1] - bounds[:, 0] > 0
    not_diff = np.invert(diff)

    values = np.atleast_2d([np.linspace(*b, num=grid_size) for b in bounds[diff]])
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
    cst_time: int,
    eps: float,
):
    """Compute log-likelihood"""

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

    sums_ll = []
    beginning_history = pd.Timestamp("1970-01-01", tz="UTC")
    one_s = pd.Timedelta("1s")
    for _, param_pair in grid_df.iterrows():
        sums_ll.append(
            log_lik(
                param_pair,
                user_df["item"],
                user_df["success"],
                (user_df["ts_reply"] - beginning_history) // one_s,  # To seconds as in pandas docs
                1 / (10000 * 60 ** 2),
                eps,
            )
        )

    return sums_ll


def main(f_results: str) -> None:
    """Get grid values, log-lik, and plot"""

    eps = np.finfo(np.float).eps

    # bounds = np.array([[0.001, 0.5], [0.00, 0.5]])
    bounds = np.array([[0.00001, 4.0], [0.00, 0.5]])
    grid_size = 20
    methods = np.array([np.logspace, np.linspace])
    grid = cp_grid_param_loglin(grid_size, bounds, methods)
    grid = pd.DataFrame(grid, columns=("alpha", "beta"))

    sns.heatmap(grid, square=True)
    plt.savefig(os.path.join("fig", "param_grid.pdf"))
    plt.close("all")

    results_df = pd.read_csv(os.path.join(paths.DATA_DIR, f_results), index_col=[0])
    results_df["ts_display"] = pd.to_datetime(
        results_df["ts_display"]
    )  # str to datetime
    results_df["ts_reply"] = pd.to_datetime(results_df["ts_reply"])  # str to datetime
    return (
        {
            user: get_all_log_lik(user_df, grid, eps)
            for user, user_df in results_df.groupby("user")
        },
        grid,
    )


# results_df = pd.read_csv(os.path.join(paths.DATA_DIR, "results.csv"), index_col=[0])


if __name__ == "__main__":
    loglikes, grid = main("results.csv")

#%%

#%%
values = loglikes.values
spam = np.meshgrid(values)

#%%
eggs = loglikes.values
#%%
plt.imshow(spam)
plt.savefig(os.path.join("fig", "param_grid.pdf"))
plt.close("all")
#%%

mean, cov = [0, 1], [(1, 0.5), (0.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
sns.jointplot(x="x", y="y", data=df)

plt.savefig(os.path.join("fig", "param_grid.pdf"))
#%%
df
#%%
tips = sns.load_dataset("tips")
sns.jointplot(x="total_bill", y="tip", data=tips, kind="reg")
plt.savefig(os.path.join("fig", "param_grid.pdf"))
#%%
flights_long = sns.load_dataset("flights")
flights = flights_long.pivot("month", "year", "passengers")

#%%
plt.close()
user_name = "aini"
logs = pd.Series(loglikes[f"{user_name}@test.com"], name="log_lik")
data = pd.concat((grid, logs), axis=1)
data = data.round(2)
data = data.pivot("alpha", "beta", "log_lik")
ax = sns.heatmap(data=data, cmap="viridis")
ax.invert_yaxis()
plt.savefig(os.path.join("fig", f"param_grid_{user_name}.pdf"))
# def enclose_log_lik_df(df, eps):
#     def log_lik_closure(param_pair):
#         return log_lik(param_pair, df["item"], df["success"], df["ts_reply"], np.ones_like(df["item"]), eps)
#     return log_lik_closure
# log_lik_cl = enclose_log_lik_df(df_, EPS)
