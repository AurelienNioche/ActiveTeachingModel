import matplotlib.pyplot as plt
import numpy as np

from . subplot import n_seen, success
from utils.plot import save_fig


def plot(
        seen, successes,
        fig_name, fig_folder,
        font_size=10, label_size=8, line_width=1,
        normalize=False,
        window=np.inf
):

    n_rows, n_cols = 2, 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6, 14))

    ax = axes[0]
    n_seen.curve(
        seen=seen,
        normalize=normalize,
        ax=ax,
        font_size=font_size,
        label_size=label_size,
        line_width=line_width * 2
    )

    ax = axes[1]
    success.curve(
        successes=successes,
        ax=ax,
        font_size=font_size,
        label_size=label_size,
        line_width=line_width * 2,
        window=window
    )

    save_fig(fig_name, fig_folder)