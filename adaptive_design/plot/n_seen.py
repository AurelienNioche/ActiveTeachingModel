import matplotlib.pyplot as plt
import numpy as np

from utils.plot import save_fig


def fig_n_seen(
        data, design_types, fig_name, fig_folder,
        y_label="N seen", colors=None):

    fig, ax = plt.subplots(figsize=(12, 6))

    if colors is None:
        colors = [f'C{i}' for i in range(len(design_types))]

    for i, dt in enumerate(design_types):

        ax.plot(data[dt], color=colors[i], label=dt)

    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)

    plt.legend()

    save_fig(fig_folder=fig_folder, fig_name=fig_name)