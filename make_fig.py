import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import explo_leitner
import numpy as np


def figure1():

    data = explo_leitner.get_data()
    data["alpha"] = data["alpha"].round(7)
    data["beta"] = data["beta"].round(4)
    data_pivoted = data.pivot("alpha", "beta", "n_learnt")

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw heatmap
    sns.heatmap(data=data_pivoted, cmap="viridis", ax=ax,
                cbar_kws={'label': 'N learnt using Leitner',
                          "orientation": "horizontal"})

    alpha_unique = data["alpha"].unique()
    beta_unique = data["beta"].unique()
    grid_size = len(alpha_unique)
    ax.set_xticks((0.5, grid_size / 2 - 0.5, grid_size-0.5))
    ax.set_yticks((0.5, grid_size / 2 - 0.5, grid_size - 0.5))

    np.random.seed(123)
    n_agent = 100

    n_learnt = data["n_learnt"].values
    alpha = data["alpha"].values
    beta = data["beta"].values

    smart_enough = np.flatnonzero(n_learnt > 0)
    slc = np.random.choice(smart_enough, size=n_agent, replace=False)
    alpha_v = alpha[slc]
    beta_v = beta[slc]
    alpha_coord = np.array([sorted(list(alpha_unique)).index(v)
                            for v in alpha_v], dtype=float) + 0.5
    beta_coord = np.array([sorted(list(beta_unique)).index(v)
                           for v in beta_v], dtype=float) + 0.5
    ax.scatter(beta_coord, alpha_coord, color='red')

    # Invert y-axis
    ax.invert_yaxis()

    # Make it look square
    ax.set_aspect(1)

    plt.tight_layout()

    fig_folder = os.path.join("fig")
    os.makedirs(fig_folder, exist_ok=True)
    plt.savefig(os.path.join(fig_folder, f"fig1_part.png"), dpi=300)


def main():


    figure1()

if __name__ == "__main__":
    main()
