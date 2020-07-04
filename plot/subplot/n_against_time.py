import numpy as np
from utils.brokenaxes import brokenaxes


def fig_n_against_time(
        gs,
        fig,
        timestamps,
        xlims,
        data, cond_labels,
        background,
        time_per_iter,
        colors,
        ylabel,
        vline=None):

    if xlims is None:
        ax = fig.add_subplot(gs)
    else:
        ax = brokenaxes(fig=fig, subplot_spec=gs, xlims=xlims)

    if vline is not None:
        ax.axvline(vline, color='red', linestyle=':', lw=4, label='Exam')

    for i, dt in enumerate(cond_labels):
        ax.plot(timestamps[i], data[i], color=colors[i], label=dt)

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)

    if xlims is None:
        y2, y1 = ax.get_ylim()
    else:
        y2, y1 = ax.get_ylim()[0]

    x = np.arange(0, background.size*time_per_iter, time_per_iter)
    ax.fill_between(x, y1, y2,
                    where=background == 1,
                    facecolor='whitesmoke',
                    edgecolor='lightgrey',
                    label='Training')
    ax.set_ylim(y2, y1)

    ax.legend(loc='upper left')
