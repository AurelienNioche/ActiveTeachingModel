import matplotlib.pyplot as plt
import matplotlib.transforms as trs
import numpy as np

from utils.plot import save_fig


def fig_p_item_seen(
        p_recall, cond_labels,
        vline=None,
        hline=None,
        background=None,
        time_per_iter=None,
        time_scale=1,
        axes=None, fig_name=None, fig_folder=None):

    if axes is None:
        n_row = len(cond_labels)
        fig, axes = plt.subplots(nrows=n_row, figsize=(5, 4*n_row))

    colors = [f'C{i}' for i in range(len(cond_labels))]

    for i, dt in enumerate(cond_labels):

        ax = axes[i]
        color = colors[i]

        if background is not None:
            trans = trs.blended_transform_factory(ax.transData,
                                                  ax.transAxes)

            x = np.arange(0, background.size * time_per_iter, time_per_iter)
            ax.fill_between(x, 0, 1,
                            where=background == 1,
                            facecolor='whitesmoke',
                            edgecolor='lightgrey',
                            transform=trans,
                            label='Training')
        if hline is not None:
            ax.axhline(hline, color='grey', linestyle='--', lw=0.5,
                       label='Learnt threshold')

        if vline is not None:
            ax.axvline(vline, color='red', linestyle=':', lw=4, label='Exam')

        line = None
        for coordinates in p_recall[i]:
            if len(coordinates):
                x, y = np.asarray(coordinates).T

                x /= time_scale
                line, = ax.plot(x, y, color=color, alpha=0.5, linewidth=0.5)

        if line is not None:
            line.set_label(dt)

        ax.set_xlabel("Time")
        ax.set_ylabel("Probability or recall")

        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])

        ax.legend(loc='lower left')

    if fig_folder is not None and fig_name is not None:
        save_fig(fig_folder=fig_folder, fig_name=fig_name)
