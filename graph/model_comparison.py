import matplotlib.pyplot as plt
import numpy as np


def bar_plot(
        means, errors, sig=None,
        title=None,
        xlabel=None, xlabel_fontsize=10, ylabel=None, ylabel_fontsize=10, xticks_fontsize=5,
        subplot_spec=None, fig=None, f_name=None, letter=None, labels=None):

    if fig is None:
        print('create_figure')
        fig = plt.figure()

    if subplot_spec:
        print('user subplot specifications')
        ax = fig.add_subplot(subplot_spec)
    else:
        print('create subplot')
        ax = fig.add_subplot(111)
    print(ax)

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


def plot_example():

    np.random.seed(123)
    means = np.random.random(size=3)
    errors = np.random.random(size=3) / 100
    sig = [(0, 1, True), (0, 2, False)]
    plot(means=means, errors=errors, sig=sig)

    plt.show()


def plot(results, color_gain, color_loss, ax):

    n = len(results.keys())

    tick_labels = [
        "Loss\nvs\ngains", "Diff. $x +$\nSame $p$", "Diff. $x -$\nSame $p$",
        "Diff. $p$\nSame $x +$", "Diff. $p$\nSame $x -$"]

    colors = ["black", color_gain, color_loss, color_gain, color_loss]
    positions = list(range(n))

    x_scatter = []
    y_scatter = []
    colors_scatter = []

    values_box_plot = []

    for i, cond in enumerate(results.keys()):

        values_box_plot.append([])

        for v in results[cond].values():
            # For box plot
            values_box_plot[-1].append(v)

            # For scatter
            y_scatter.append(v)
            x_scatter.append(i)
            colors_scatter.append(colors[i])

    fontsize = 10

    ax.scatter(x_scatter, y_scatter, c=colors_scatter, s=30, alpha=0.5, linewidth=0.0, zorder=1)

    ax.axhline(0.5, linestyle='--', color='0.3', zorder=-10, linewidth=0.5)

    ax.set_yticks(np.arange(0.4, 1.1, 0.2))

    ax.tick_params(axis='both', labelsize=fontsize)

    # ax.set_xlabel("Type of control\nMonkey {}.".format(monkey), fontsize=fontsize)
    # ax.set_xlabel("Control type", fontsize=fontsize)
    ax.set_ylabel("Success rate", fontsize=fontsize)

    ax.set_ylim(0.35, 1.02)

    # Boxplot
    bp = ax.boxplot(values_box_plot, positions=positions, labels=tick_labels, showfliers=False, zorder=2)

    for e in ['boxes', 'caps', 'whiskers', 'medians']:  # Warning: only one box, but several whiskers by plot
        for b in bp[e]:
            b.set(color='black')
            # b.set_alpha(1)

    ax.set_aspect(3)


if __name__ == "__main__":

    plot_example()
