import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
import numpy as np

from plot.generic import save_fig


def curve(p_recall, fig_name='n_learnt.pdf',
          font_size=12, line_width=2,
          label_size=8, threshold=0.95, ax=None):

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

    # Data pre-processing
    n_item, n_iteration = p_recall.shape

    learnt = p_recall[:] > threshold
    n_learnt = np.sum(learnt, axis=0)
    y = n_learnt / n_item

    # Horizontal lines
    ax.axhline(0.5, linewidth=0.5, linestyle='dotted', color='black', alpha=0.5)
    ax.axhline(0.25, linewidth=0.5, linestyle='dotted', color='black', alpha=0.5)
    ax.axhline(0.75, linewidth=0.5, linestyle='dotted', color='black', alpha=0.5)

    # Plot
    ax.plot(y, color="C0", linewidth=line_width)

    # Both axis
    ax.tick_params(axis="both", labelsize=label_size)

    # x-axis
    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_xlim(1, n_iteration)
    ax.set_xticks((1, int(n_iteration/2), n_iteration))

    # y-axis
    ax.set_ylabel(f'N | $p_{{recall}} > {threshold}$', fontsize=font_size)
    ax.set_ylim((-0.01, 1.01))
    ax.set_yticks((0, 0.5, 1))
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if ax is None:
        save_fig(fig_name=fig_name)
