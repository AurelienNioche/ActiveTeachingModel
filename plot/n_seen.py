import numpy as np
import matplotlib.pyplot as plt

from plot.generic import save_fig


def curve(seen, fig_name='seen.pdf',
          font_size=42, line_width=3,
          label_size=22):

    y = np.sum(seen, axis=0)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_ylabel('N seen', fontsize=font_size)
    ax.set_xlabel('Time', fontsize=font_size)
    # ax.set_yticks((0, 1))
    # ax.set_ylim((0, 1))
    ax.tick_params(axis="both", labelsize=label_size)
    ax.plot(y, color="black", linewidth=line_width)

    plt.tight_layout()

    save_fig(fig_name)
