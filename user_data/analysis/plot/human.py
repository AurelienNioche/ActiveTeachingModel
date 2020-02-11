import matplotlib.pyplot as plt
import numpy as np

from . subplot import n_seen as subplot_n_seen
from .subplot import success as subplot_success
from utils.plot import save_fig


def plot(
        n_seen, success,
        fig_name, fig_folder,
        font_size=10, label_size=8, line_width=1,
        normalize_by=None,
        window=np.inf
):

    n_rows, n_cols = 2, 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6, 14))

    ax = axes[0]
    subplot_n_seen.curve(
        y=n_seen,
        normalize_by=normalize_by,
        ax=ax,
        font_size=font_size,
        label_size=label_size,
        line_width=line_width * 2
    )

    ax = axes[1]
    subplot_success.curve(
        successes=success,
        ax=ax,
        font_size=font_size,
        label_size=label_size,
        line_width=line_width * 2,
        window=window
    )

    save_fig(fig_name, fig_folder)
