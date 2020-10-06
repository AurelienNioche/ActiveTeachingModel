import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

import numpy as np
import math
from string import ascii_uppercase

import analyse_s
import explo_leitner
import analyse_s_p_recall


def roundup(x):
    return int(math.ceil(x / 100.0)) * 100


def roundup10(x):
    return int(math.ceil(x / 10.0)) * 10


def rounddown10(x):
    return int(math.floor(x / 10.0)) * 10


def rounduptenth(x):
    return math.ceil(x * 10.0) / 10


def rounddowntenth(x):
    return math.floor(x * 10.0) / 10


def heatmap(data, ax):
    data["alpha"] = data["alpha"].round(7)
    data["beta"] = data["beta"].round(4)
    data_pivoted = data.pivot("alpha", "beta", "n_learnt")

    # Draw heatmap
    sns.heatmap(data=data_pivoted, cmap="viridis", ax=ax,
                cbar_kws={'label': 'N learned using Leitner',
                          "orientation": "horizontal"})

    alpha_unique = sorted(data["alpha"].unique())
    beta_unique = sorted(data["beta"].unique())
    grid_size = len(alpha_unique)

    ax.set_ylabel(r"$\alpha$")
    ax.set_xlabel(r"$\beta$")

    ax.set_xticks((0.5, grid_size / 2 - 0.5, grid_size - 0.5))
    ax.set_xticklabels((beta_unique[0], beta_unique[grid_size//2 -1],
                        beta_unique[-1]), rotation=0)

    ax.set_yticks((0.5, grid_size / 2 - 0.5, grid_size - 0.5))
    ax.set_yticklabels((alpha_unique[0], alpha_unique[grid_size//2 - 1],
                        alpha_unique[-1]), rotation=0)

    np.random.seed(123)
    n_agent = 100

    n_learnt = data["n_learnt"].values
    alpha = data["alpha"].values
    beta = data["beta"].values

    smart_enough = np.flatnonzero(n_learnt > 0)
    slc = np.random.choice(smart_enough, size=n_agent, replace=False)
    alpha_v = alpha[slc]
    beta_v = beta[slc]
    alpha_coord = np.array([alpha_unique.index(v)
                            for v in alpha_v], dtype=float) + 0.5
    beta_coord = np.array([beta_unique.index(v)
                           for v in beta_v], dtype=float) + 0.5
    ax.scatter(beta_coord, alpha_coord, color='red', s=10)

    # Invert y-axis
    ax.invert_yaxis()

    # Make it look square
    ax.set_aspect(1)


def rename_teachers(data):
    dic = {
        "leitner": "Leitner",
        "forward": "Conservative\nSampling",
        "threshold": "Myopic",
    }

    for k, v in dic.items():
        data["teacher"] = data["teacher"].replace([k], v)
    return data


def boxplot_n_learnt(data, ax,
                     x_label="Teacher",
                     y_label="N learned", dot_size=3, dot_alpha=0.7):

    data = rename_teachers(data)
    data = data.rename(columns={
        "n_learnt": y_label,
        "teacher": x_label
    })

    order = ["Leitner", "Myopic", "Conservative\nSampling"]
    colors = ["C0", "C1", "C2"]

    sns.boxplot(x=x_label, y=y_label, data=data, ax=ax,
                palette=colors, order=order,
                showfliers=False)
    sns.stripplot(x=x_label, y=y_label, data=data, s=dot_size,
                  color="0.25", alpha=dot_alpha, ax=ax, order=order)

    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=13)

    ymax = roundup(np.max(data[y_label]))
    ax.set_ylim(-0.1, ymax+0.1)
    ax.set_yticks((0, ymax//2, ymax))

    ax.set_xlabel("")


def boxplot_n_learnt_n_seen(data, ax,
                            x_label="Teacher",
                            y_label="N learned / N seen",
                            dot_size=3, dot_alpha=0.7):

    data = rename_teachers(data)
    data = data.rename(columns={
        "teacher": x_label,
        "ratio_n_learnt_n_seen": y_label
    })

    order = ["Leitner", "Myopic", "Conservative\nSampling"]
    colors = ["C0", "C1", "C2"]

    sns.boxplot(x="Teacher", y=y_label, data=data, ax=ax,
                palette=colors, order=order,
                showfliers=False)
    sns.stripplot(x="Teacher", y=y_label, data=data,
                  color="0.25", alpha=dot_alpha, ax=ax, order=order,
                  s=dot_size)

    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=13)

    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks((0, 0.5, 1))

    ax.set_xlabel("")


def prediction_error(data, ax, title, color,
                     x_label="Time",
                     y_label="Prediction error\n"
                             "(probability of recall)"):

    data = data.rename(columns={
        "time": x_label,
        "p_err_mean": y_label
    })

    sns.lineplot(data=data, x=x_label, y=y_label, ci="sd", ax=ax, color=color)

    ax.set_ylim(0, 1)
    ax.set_title(title)


def figure2():

    dataset = "n_learnt_leitner"

    df_heat = explo_leitner.get_data()

    df_omni = analyse_s.get_data(dataset_name=dataset,
                                 condition_name="Nspec-omni")

    df_not_omni = analyse_s.get_data(dataset_name=dataset,
                                     condition_name="Nspec-Nomni")

    err_threshold = analyse_s_p_recall.get_data(
        dataset_name=dataset,
        condition_name="Nspec-Nomni",
        teacher_name="threshold")

    err_forward = analyse_s_p_recall.get_data(
        dataset_name=dataset,
        condition_name="Nspec-Nomni",
        teacher_name="forward")

    fig = plt.figure(figsize=(13, 8.5))
    fig.suptitle("Artificial: Non item specific",
                 fontweight="bold",
                 fontsize=17)

    # Thanks to the Matplotlib creators that I love
    pos = [(4, 3, (1, 4)), (4, 3, (2, 5)), (4, 3, (3, 6)),
           (4, 3, 7), (4, 3, 10), (4, 3, (8, 11)), (4, 3, (9, 12))]
    axes = [fig.add_subplot(*p) for p in pos]

    heatmap(data=df_heat, ax=axes[0])
    axes[0].text(-0.6, 1.05, ascii_uppercase[0],
                 transform=axes[0].transAxes, size=15, weight='bold')

    boxplot_n_learnt(data=df_omni, ax=axes[1])
    axes[1].set_title("N learned\n", fontstyle='italic', fontsize=15)
    axes[1].text(-0.2, 1.05, ascii_uppercase[1],
                 transform=axes[1].transAxes, size=15, weight='bold')

    axes[1].text(-0.25, 0.5, "Omniscient",
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=axes[1].transAxes, size=15, weight='bold', rotation=90, style="italic")

    boxplot_n_learnt_n_seen(data=df_omni, ax=axes[2])
    axes[2].set_title("N learned / N seen\n", fontstyle='italic', fontsize=15)
    axes[2].text(-0.2, 1.05, ascii_uppercase[2],
                 transform=axes[2].transAxes, size=15, weight='bold')

    prediction_error(data=err_threshold, ax=axes[3],
                     title="Myopic",
                     color="C1")
    axes[3].text(-0.25, 1.15, ascii_uppercase[3],
                 transform=axes[3].transAxes, size=15, weight='bold')

    ax = fig.add_subplot(2, 3, 4)
    ax.set_axis_off()
    ax.text(-0.35, 0.5, "Non omniscient",
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes, size=15, weight='bold',
            rotation=90, style="italic")

    prediction_error(data=err_forward, ax=axes[4],
                     title="Conservative Sampling",
                     color="C2")

    boxplot_n_learnt(data=df_not_omni, ax=axes[5])
    axes[5].text(-0.2, 1.05, ascii_uppercase[4],
                 transform=axes[5].transAxes, size=15, weight='bold')

    boxplot_n_learnt_n_seen(data=df_not_omni, ax=axes[6])
    axes[6].text(-0.2, 1.05, ascii_uppercase[5],
                 transform=axes[6].transAxes, size=15, weight='bold')

    plt.tight_layout(rect=[-0.05, 0, 1, 1])

    # plt.show()

    fig_folder = os.path.join("fig")
    os.makedirs(fig_folder, exist_ok=True)
    plt.savefig(os.path.join(fig_folder, f"fig2.png"), dpi=300)
    plt.savefig(os.path.join(fig_folder, f"fig2.pdf"))


def figure3():

    dataset = "n_learnt_leitner"

    df_omni = analyse_s.get_data(dataset_name=dataset,
                                 condition_name="spec-omni")

    df_not_omni = analyse_s.get_data(dataset_name=dataset,
                                     condition_name="spec-Nomni")

    err_threshold = analyse_s_p_recall.get_data(
        dataset_name=dataset,
        condition_name="spec-Nomni",
        teacher_name="threshold")

    err_forward = analyse_s_p_recall.get_data(
        dataset_name=dataset,
        condition_name="spec-Nomni",
        teacher_name="forward")

    fig = plt.figure(figsize=(13, 8.5))
    fig.suptitle("Artificial: Item specific",
                 fontweight="bold",
                 fontsize=17)

    # Thanks to the Matplotlib creators that I love
    pos = [(4, 3, (2, 5)), (4, 3, (3, 6)),
           (4, 3, 7), (4, 3, 10), (4, 3, (8, 11)), (4, 3, (9, 12))]
    axes = [fig.add_subplot(*p) for p in pos]

    boxplot_n_learnt(data=df_omni, ax=axes[0])
    axes[0].set_title("N learned\n", fontstyle='italic', fontsize=15)
    axes[0].text(-0.2, 1.05, ascii_uppercase[0],
                 transform=axes[0].transAxes, size=15, weight='bold')

    axes[0].text(-0.25, 0.5, "Omniscient",
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=axes[0].transAxes, size=15, weight='bold',
                 rotation=90, style="italic")

    boxplot_n_learnt_n_seen(data=df_omni, ax=axes[1])
    axes[1].set_title("N learned / N seen\n", fontstyle='italic', fontsize=15)
    axes[1].text(-0.2, 1.05, ascii_uppercase[1],
                 transform=axes[1].transAxes, size=15, weight='bold')

    prediction_error(data=err_threshold, ax=axes[2],
                     title="Myopic",
                     color="C1")
    axes[2].text(-0.25, 1.15, ascii_uppercase[2],
                 transform=axes[2].transAxes, size=15, weight='bold')

    ax = fig.add_subplot(2, 3, 4)
    ax.set_axis_off()
    ax.text(-0.35, 0.5, "Non omniscient",
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes, size=15, weight='bold',
            rotation=90, style="italic")

    prediction_error(data=err_forward, ax=axes[3],
                     title="Conservative Sampling",
                     color="C2")

    boxplot_n_learnt(data=df_not_omni, ax=axes[4])
    axes[4].text(-0.2, 1.05, ascii_uppercase[3],
                 transform=axes[4].transAxes, size=15, weight='bold')

    boxplot_n_learnt_n_seen(data=df_not_omni, ax=axes[5])
    axes[5].text(-0.2, 1.05, ascii_uppercase[4],
                 transform=axes[5].transAxes, size=15, weight='bold')

    plt.tight_layout(rect=[-0.05, 0, 1, 1])

    fig_folder = os.path.join("fig")
    os.makedirs(fig_folder, exist_ok=True)
    plt.savefig(os.path.join(fig_folder, f"fig3.png"), dpi=300)
    plt.savefig(os.path.join(fig_folder, f"fig3.pdf"))


def make_statistics_artificial():
    # Artificial --------------- #
    print("Artificial" + " " + "*"*50)

    dataset = "n_learnt_leitner"

    for condition in "Nspec-omni", "Nspec-Nomni", "spec-omni", "spec-Nomni":

        df = analyse_s.get_data(dataset_name=dataset,
                                condition_name=condition)
        print(condition)

        print(f"n item learnt:")
        for teacher in ('threshold', 'forward'):
            u, p = stats.mannwhitneyu(
                x=df[df.teacher == teacher]["n_learnt"],
                y=df[df.teacher == "leitner"]["n_learnt"],
                alternative='two-sided', use_continuity=False)
            print(f"{teacher} // Leitner")
            p_f = f"$p={p:.3f}$" if p >= 0.001 else "$p<0.001$"
            print(f"$u={u}$; {p_f}")

        print()
        print(f"n learnt / n seen:")
        for teacher in ('threshold', 'forward'):
            u, p = stats.mannwhitneyu(
                x=df[df.teacher == teacher]["ratio_n_learnt_n_seen"],
                y=df[df.teacher == "leitner"]["ratio_n_learnt_n_seen"],
                alternative='two-sided', use_continuity=True)
            print(f"{teacher} // Leitner")
            p_f = f"$p={p:.3f}$" if p >= 0.001 else "$p<0.001$"
            print(f"$u={u}$; {p_f}")
        print()


def main():

    make_statistics_artificial()
    figure2()
    figure3()


if __name__ == "__main__":
    main()
