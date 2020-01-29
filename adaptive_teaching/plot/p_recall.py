import matplotlib.pyplot as plt
import numpy as np

from utils.plot import save_fig


def fig_p_recall(data, condition_labels, fig_name=None, fig_folder=None,
                 y_label="Probability of recall", colors=None):

    fig, ax = plt.subplots(figsize=(5, 4))

    if colors is None:
        colors = [f'C{i}' for i in range(len(condition_labels))]

    for i, lb in enumerate(condition_labels):

        if isinstance(data[lb], list):
            n_trial = len(data[lb])
            means = np.asarray([np.mean(data[lb][i]) for i in range(n_trial)])
            sds = np.asarray([np.std(data[lb][i]) for i in range(n_trial)])

        else:
            n_trial = data[lb].shape[1]
            means = np.mean(data[lb], axis=0)
            sds = np.std(data[lb], axis=0)

        ax.plot(means, color=colors[i], label=lb)
        ax.fill_between(range(n_trial),
                        means-sds,
                        means+sds,
                        alpha=.2, color=colors[i])

        # ax.set_xlim(0, len(means))
        # ax.set_ylim(0, 1)

    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)

    plt.legend(loc='lower left')

    if fig_folder is not None and fig_name is not None:
        save_fig(fig_folder=fig_folder, fig_name=fig_name)
    else:
        plt.show()


def fig_p_recall_item(p_recall, condition_labels, fig_name=None,
                      fig_folder=None):

    n_row = len(condition_labels)
    fig, axes = plt.subplots(nrows=n_row, figsize=(5, 4*n_row))

    colors = [f'C{i}' for i in range(len(condition_labels))]

    for i, dt in enumerate(condition_labels):

        ax = axes[i]
        color = colors[i]

        n_trial = p_recall[dt].shape[1]

        mean = np.mean(p_recall[dt], axis=0)
        std = np.std(p_recall[dt], axis=0)

        ax.plot(mean, color=color, label=dt)
        ax.fill_between(range(n_trial),
                        mean-std,
                        mean+std,
                        alpha=.1, color=color)
        ys = p_recall[dt]

        for y in ys:
            ax.plot(y, color=color, alpha=0.5, linewidth=0.5)
            x_ticks = np.zeros(3, dtype=int)
            x_ticks[:] = np.linspace(0, len(y), 3)

            ax.set_xticks(x_ticks)

        ax.set_xlabel("Time")
        ax.set_ylabel("Probability or recall")

        ax.legend(loc='lower left')

    if fig_folder is not None and fig_name is not None:
        save_fig(fig_folder=fig_folder, fig_name=fig_name)
    else:
        plt.show()
