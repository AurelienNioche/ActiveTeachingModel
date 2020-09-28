import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

FIG_FOLDER = os.path.join('fig', "param_recovery_gen_param")
os.makedirs(FIG_FOLDER, exist_ok=True)


def plot_param_space(grid: pd.DataFrame,
                     values: np.ndarray,
                     f_name: str) -> None:
    """Heatmap of the alpha-beta parameter space"""

    print("Plotting heatmap...")
    plt.close()
    values = pd.Series(values, name="values")
    data = pd.concat((grid, values), axis=1)

    try:  # Duplicated entries can appear with rounding
        data = data.round(4).pivot("alpha", "beta", "values")
    except:
        data = data.pivot("alpha", "beta", "values")
    ax = sns.heatmap(data=data, cmap="viridis")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_FOLDER,
                             f"{f_name}.png"), dpi=300)
    print("Done!")


def main():

    grid_df = pd.read_csv("grid.csv", index_col=[0])
    ll_df = pd.read_csv("lls.csv", index_col=[0])

    lls = np.sum(ll_df.values, axis=1)

    plot_param_space(grid=grid_df, values=lls, f_name="lls")

    p = (lls-min(lls))/(max(lls)-min(lls))
    p /= np.sum(p)

    # grid = grid_df.values
    # idx = np.random.choice(np.arange(len(grid)), p=p)
    # print(grid[idx])

    plot_param_space(grid=grid_df, values=p, f_name="p")


if __name__ == "__main__":
    main()
