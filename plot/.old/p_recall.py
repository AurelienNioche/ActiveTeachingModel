import matplotlib.pyplot as plt
import matplotlib.transforms as trs
import numpy as np

from utils.brokenaxes import brokenaxes


def fig_p_item_seen(
    fig,
    gs,
    xlims,
    p_recall,
    label,
    background,
    time_per_iter,
    color,
    vline=None,
    hline=None,
):

    if xlims is None:
        ax = fig.add_subplot(gs)
    else:
        ax = brokenaxes(fig=fig, subplot_spec=gs, xlims=xlims)

    if hline is not None:
        ax.axhline(
            hline, color="grey", linestyle="--", lw=0.5, label="Learnt threshold"
        )

    if vline is not None:
        ax.axvline(vline, color="red", linestyle=":", lw=4, label="Exam")

    lines = None
    for coordinates in p_recall:
        if len(coordinates):
            x, y = np.asarray(coordinates).T
            lines = ax.plot(x, y, color=color, alpha=0.5, linewidth=0.5)

    if lines is not None:
        if xlims is None:
            line = lines[0]
        else:
            line = lines[0][0]
        line.set_label(label)

    ax.set_xlabel("Time")
    ax.set_ylabel("Probability or recall")

    ax.set_ylim(-0.005, 1.005)
    ax.set_yticks([0, 0.5, 1])

    ax.legend(loc="lower left")

    if xlims is None:
        y2, y1 = ax.get_ylim()
    else:
        y2, y1 = ax.get_ylim()[0]

    x = np.arange(0, background.size * time_per_iter, time_per_iter)
    ax.fill_between(
        x,
        y1,
        y2,
        where=background == 1,
        facecolor="whitesmoke",
        edgecolor="lightgrey",
        label="Training",
    )
    ax.set_ylim(y2, y1)
