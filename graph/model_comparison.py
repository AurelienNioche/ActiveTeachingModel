import matplotlib.pyplot as plt
import numpy as np
import os

FIG_FOLDER = "fig"


def bar_plot(
        means, errors, sig=None,
        title=None,
        xlabel=None, xlabel_fontsize=10, ylabel=None, ylabel_fontsize=10, xticks_fontsize=5,
        subplot_spec=None, fig=None, f_name=None, letter=None, labels=None):

    if fig is None:
        fig = plt.figure()

    if subplot_spec:
        ax = fig.add_subplot(subplot_spec)

    else:
        ax = fig.add_subplot(111)

    # If adding a letter
    if letter:
        ax.text(
            s=letter, x=-0.1, y=-0.68, horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes,
            fontsize=20)

    if labels is None:
        labels = [str(i + 1) for i in range(len(means))]

    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', length=0, labelsize=xticks_fontsize)

    # Set x labels
    labels_pos = np.arange(len(labels))
    ax.set_xticklabels(labels)
    ax.set_xticks(labels_pos)

    # For significance bars
    if sig:

        y_inc_line = 0.05
        y_inc_text = 0.07

        shift = 0.11

        for idx, (i, j, k) in enumerate(sig):

            x = (i+j)/2
            y = max(means[i:j+1])

            ax.hlines(y=y + y_inc_line + shift*idx,
                      xmin=i, xmax=j, color='black')

            if k < 0.001:
                s = '***'
            elif k < 0.01:
                s = '**'
            elif k < 0.05:
                s = '*'
            else:
                s = '$^{ns}$'

            ax.text(s=s,
                    x=x,
                    y=y + y_inc_text + shift*idx,
                    horizontalalignment='center', verticalalignment='center')

    ax.set_ylim(0, 1)

    ax.set_yticks((0, 0.5, 1))

    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)

    ax.set_title(title)

    # Create
    ax.bar(labels_pos, means, yerr=errors, edgecolor="white", align="center", color="grey")

    if f_name is not None:
        plt.savefig(f_name)


def bar_example():

    np.random.seed(123)
    means = np.random.random(size=3)
    errors = np.random.random(size=3) / 100
    sig = [(0, 1, True), (0, 2, False)]
    bar_plot(means=means, errors=errors, sig=sig)

    plt.show()


def scatter_plot(
        data_list, colors=None, x_tick_labels=None, fontsize=10,
        subplot_spec=None, fig=None, f_name=None, letter=None, y_lim=None, y_label=None,
        h_line=None, invert_y_axis=False
):

    if fig is None:
        fig = plt.figure()

    if subplot_spec:
        ax = fig.add_subplot(subplot_spec)

    else:
        ax = fig.add_subplot(111)

    # If adding a letter
    if letter:
        ax.text(
            s=letter, x=-0.1, y=-0.68, horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes,
            fontsize=20)

    n = len(data_list)

    # Colors
    if colors is None:
        colors = ["black" for _ in range(n)]

    positions = list(range(n))

    x_scatter = []
    y_scatter = []
    colors_scatter = []

    values_box_plot = data_list

    # For scatter
    for i, data in enumerate(data_list):
        for v in data:
            y_scatter.append(v)
            x_scatter.append(i)
            colors_scatter.append(colors[i])

    ax.scatter(x_scatter, y_scatter, c=colors_scatter, s=30, alpha=0.5, linewidth=0.0, zorder=1)

    if h_line:
        ax.axhline(h_line, linestyle='--', color='0.3', zorder=-10, linewidth=0.5)

    ax.tick_params(axis='both', labelsize=fontsize)

    # ax.set_xlabel("Type of control\nMonkey {}.".format(monkey), fontsize=fontsize)
    # ax.set_xlabel("Control type", fontsize=fontsize)
    if y_label:
        ax.set_ylabel(y_label, fontsize=fontsize)

    # ax.set_yticks(np.arange(0.4, 1.1, 0.2))

    if y_lim:
        ax.set_ylim(y_lim)

    if invert_y_axis:
        ax.invert_yaxis()

    if x_tick_labels is None:
        x_tick_labels = ["" for _ in range(n)]

    # Boxplot
    bp = ax.boxplot(values_box_plot, positions=positions, labels=x_tick_labels, showfliers=False, zorder=2)

    for e in ['boxes', 'caps', 'whiskers', 'medians']:  # Warning: only one box, but several whiskers by plot
        for b in bp[e]:
            b.set(color='black')
            # b.set_alpha(1)

    # ax.set_aspect(3)
    plt.tight_layout()

    if f_name is not None:
        os.makedirs(FIG_FOLDER, exist_ok=True)
        plt.savefig(f'{FIG_FOLDER}/{f_name}')

    else:
        plt.show()


def scatter_example():

    mu, sigma = 0, 0.1  # mean and standard deviation
    n = 50
    data_list = [np.random.normal(mu, sigma, size=n), np.random.poisson(lam=1, size=n)]
    scatter_plot(data_list=data_list)


if __name__ == "__main__":

    scatter_example()
