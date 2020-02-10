import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils.plot import save_fig


def curve(seen, fig_name=None,
          font_size=12, line_width=3,
          label_size=8, ax=None, normalize=False):

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

    # Data pre-processing
    n_item, n_iteration = seen.shape

    x = np.arange(n_iteration) + 1
    y = np.sum(seen, axis=0)

    if normalize:
        y[:] = y/n_item

    # Plot
    ax.plot(x, y, color='C0', linewidth=line_width)

    # Both axis
    ax.tick_params(axis="both", labelsize=label_size)

    # x-axis
    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_xlim(1, n_iteration)
    ax.set_xticks((1,
                   int(n_iteration * 0.25),
                   int(n_iteration * 0.5),
                   int(n_iteration * 0.75),
                   n_iteration))

    # y-axis
    ax.set_ylabel('$N_{seen}$', fontsize=font_size)

    if normalize:  # Scale is normalized
        ax.set_ylim((-0.01, 1.01))
        ax.set_yticks((0, 0.5, 1))

        # Horizontal lines
        ax.axhline(0.5, linewidth=0.5, linestyle='dotted',
                   color='black', alpha=0.5)
        ax.axhline(0.25, linewidth=0.5, linestyle='dotted',
                   color='black', alpha=0.5)
        ax.axhline(0.75, linewidth=0.5, linestyle='dotted',
                   color='black', alpha=0.5)

    else:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if fig_name is not None:
        save_fig(fig_name=fig_name)
