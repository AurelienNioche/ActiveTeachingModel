import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.preprocessing import normalize

FIG_FOLDER = os.path.join('fig', "param_recovery_gen_param_best_likelihood")
os.makedirs(FIG_FOLDER, exist_ok=True)


def plot_param_space(grid: pd.DataFrame,
                     lls: np.ndarray,
                     best_p: pd.DataFrame) -> None:
    """Heatmap of the alpha-beta parameter space"""

    print("Plotting heatmap...")
    plt.close()

    lls = pd.Series(lls, name="lls")
    data = pd.concat((grid, lls), axis=1)

    try:  # Duplicated entries can appear with rounding
        data = data.round(4).pivot("alpha", "beta", "lls")
    except:
        data = data.pivot("alpha", "beta", "lls")

    fig, ax = plt.subplots()
    sns.heatmap(data=data, cmap="viridis", ax=ax)
    ax.invert_yaxis()

    sns.scatterplot(data=best_p, x="beta", y="alpha", color="red")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_FOLDER,
                             f"best_param.png"), dpi=300)
    print("Done!")


def main():

    grid_df = pd.read_csv("grid.csv", index_col=[0])
    ll_df = pd.read_csv("lls.csv", index_col=[0])

    print(ll_df.values)

    n = len(ll_df.columns)

    best_p = np.zeros((n, 2))

    grid = np.array(grid_df.values)
    grid_size = int(np.sqrt(grid.shape[0]))

    yy, xx = np.meshgrid(np.arange(grid_size), np.arange(grid_size),
                         indexing='ij')
    y_f = yy.flatten()
    x_f = xx.flatten()

    a = np.zeros(n)
    b = np.zeros(n)

    for i, c in enumerate(ll_df.columns):

        best_idx = ll_df[c].idxmax()
        max_v = grid[best_idx]
        best_p[i, :] = max_v

        alpha_idx = y_f[np.argwhere(grid[:, 0] == max_v[0])[0][0]]
        beta_idx = x_f[np.argwhere(grid[:, 1] == max_v[1])[0][0]]

        a[i] = alpha_idx
        b[i] = beta_idx

    best_p_df = pd.DataFrame(best_p, columns=("alpha", "beta"))
    best_p_df.to_csv("best_param.csv")

    lls = np.sum(ll_df.values, axis=1)
    plot_param_space(grid=grid_df, lls=lls, best_p=pd.DataFrame({"alpha": a + 0.5, "beta": b+0.5}))


if __name__ == "__main__":
    main()
