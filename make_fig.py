import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from string import ascii_uppercase

import analyse_s
import explo_leitner
import analyse_s_p_recall


def heatmap(data, ax):
    data["alpha"] = data["alpha"].round(7)
    data["beta"] = data["beta"].round(4)
    data_pivoted = data.pivot("alpha", "beta", "n_learnt")

    # Draw heatmap
    sns.heatmap(data=data_pivoted, cmap="viridis", ax=ax,
                cbar_kws={'label': 'N learnt using Leitner',
                          "orientation": "horizontal"})

    alpha_unique = data["alpha"].unique()
    beta_unique = data["beta"].unique()
    grid_size = len(alpha_unique)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    ax.set_xticks((0.5, grid_size / 2 - 0.5, grid_size - 0.5))
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

    # Add letter

    ax.text(-0.1, 1.1, ascii_uppercase[0],
            transform=ax.transAxes, size=20, weight='bold')


def rename_teachers(data):
    dic = {
        "leitner": "Leitner",
        "forward": "Conservative\nSampling",
        "threshold": "Myopic",
    }

    for k, v in dic.items():
        data["Teacher"] = data["Teacher"].replace([k], v)
    return data


def boxplot_n_learnt(data, ax, y_label="N learned"):

    data = data.rename(columns={
        "Items learnt one day later": y_label
    })

    data = rename_teachers(data)

    order = ["Leitner", "Myopic", "Conservative\nSampling"]
    colors = ["C0", "C1", "C2"]

    sns.boxplot(x="Teacher", y=y_label, data=data, ax=ax,
                palette=colors, order=order)
    sns.stripplot(x="Teacher", y=y_label, data=data,
                  color="0.25", alpha=0.7, ax=ax, order=order)

    ax.set_ylim(0, 501)
    ax.set_yticks((0, 250, 500))


def boxplot_n_learnt_n_seen(data, ax, y_label="N learned / N seen"):

    data = data.rename(columns={
        "N learnt / N seen": y_label
    })

    data = rename_teachers(data)

    order = ["Leitner", "Myopic", "Conservative\nSampling"]
    colors = ["C0", "C1", "C2"]

    sns.boxplot(x="Teacher", y=y_label, data=data, ax=ax,
                palette=colors, order=order)
    sns.stripplot(x="Teacher", y=y_label, data=data,
                  color="0.25", alpha=0.7, ax=ax, order=order)

    ax.set_ylim(0, 1)
    ax.set_yticks((0, 0.5, 1))


def prediction_error(data, ax, title,
                     x_label="Time",
                     y_label="Prediction error\n"
                             "(probability of recall)"):

    data = data.rename(columns={
        "time": x_label,
        "p_err_mean": y_label
    })

    sns.lineplot(data=data, x=x_label, y=y_label, ci="sd", ax=ax)

    ax.set_ylim(0, 1)
    ax.set_title(title)


def figure2():

    df_heat = explo_leitner.get_data()

    df_omni = analyse_s.get_data(trial_name="explo_leitner_geolin",
                                 condition_name="Nspec-omni")

    df_not_omni = analyse_s.get_data(trial_name="explo_leitner_geolin",
                                     condition_name="Nspec-Nomni")

    err_threshold = analyse_s_p_recall.get_data(
        trial_name="explo_leitner_geolin",
        condition_name="Nspec-Nomni",
        teacher_name="threshold")

    err_forward = analyse_s_p_recall.get_data(
        trial_name="explo_leitner_geolin",
        condition_name="Nspec-Nomni",
        teacher_name="forward")

    fig = plt.figure(figsize=(14, 10))

    # Thanks to the Matplotlib creators that I love
    pos = [(4, 3, (1, 4)), (4, 3, (2, 5)), (4, 3, (3, 6)),
           (4, 3, 7), (4, 3, 10), (4, 3, (8, 11)), (4, 3, (9, 12))]
    axes = [fig.add_subplot(*p) for p in pos]

    # Create figure and axes
    # fig, ax = plt.subplots(figsize=(6, 6))
    heatmap(data=df_heat, ax=axes[0])
    boxplot_n_learnt(data=df_omni, ax=axes[1])
    boxplot_n_learnt_n_seen(data=df_omni, ax=axes[2])

    prediction_error(data=err_threshold, ax=axes[3],
                     title="Myopic")
    prediction_error(data=err_forward, ax=axes[4],
                     title="Conservative Sampling")

    boxplot_n_learnt(data=df_not_omni, ax=axes[5])
    boxplot_n_learnt_n_seen(data=df_not_omni, ax=axes[6])

    # print("Saving fig...")
    # fig.savefig(fig_path, dpi=300)

    # df = explo_leitner.get_data()
    # heatmap(data=df, ax=ax)

    plt.tight_layout()
    plt.show()

    fig_folder = os.path.join("fig")
    os.makedirs(fig_folder, exist_ok=True)
    plt.savefig(os.path.join(fig_folder, f"fig1_part.png"), dpi=300)


def figure3():

    fig = plt.figure(figsize=(6, 4))

    # Thanks to the Matplotlib creators that I love
    pos = [(4, 3, (2, 5)), (4, 3, (3, 6)),
           (4, 3, 7), (4, 3, 10), (4, 3, (8, 11)), (4, 3, (9, 12))]
    axes = [fig.add_subplot(*p) for p in pos]

    plt.tight_layout()
    plt.show()


def main():

    figure2()

if __name__ == "__main__":
    main()
