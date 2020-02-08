import matplotlib.pyplot as plt
# import numpy as np

from utils.plot import save_fig


def fig_n_seen(
        data, condition_labels,
        ax=None,
        fig_name=None, fig_folder=None,
        y_label=None, colors=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    if colors is None:
        colors = [f'C{i}' for i in range(len(condition_labels))]

    for i, dt in enumerate(condition_labels):

        ax.plot(data[dt], color=colors[i], label=dt)

    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)

    plt.legend()

    if fig_folder is not None and fig_name is not None:
        save_fig(fig_folder=fig_folder, fig_name=fig_name)
    # else:
    #     plt.show()
