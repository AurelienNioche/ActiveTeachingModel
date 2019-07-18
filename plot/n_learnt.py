import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot.generic import save_fig


def curve(p_recall, fig_name='n_learnt.pdf',
          font_size=42, line_width=3,
          label_size=22, threshold=0.95, ax=None):

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

    # Data pre-processing
    learnt = p_recall[:] > threshold
    n_item = p_recall.shape[0]
    n_learnt = np.sum(learnt, axis=0)
    y = n_learnt / n_item

    ax.plot(y, color="black", linewidth=line_width)

    ax.set_xlabel('Time', fontsize=font_size)

    ax.tick_params(axis="both", labelsize=label_size)

    ax.set_yticks((0, 0.25, 0.5, 0.75, 1))
    ax.set_ylim((0, 1))
    ax.set_ylabel('N learnt', fontsize=font_size)
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.axhline(0.5, linewidth=1, linestyle='--', color='black', alpha=0.5)
    ax.axhline(0.25, linewidth=1, linestyle='--', color='black', alpha=0.5)
    ax.axhline(0.75, linewidth=1, linestyle='--', color='black', alpha=0.5)

    if ax is None:
        save_fig(fig_name=fig_name)
