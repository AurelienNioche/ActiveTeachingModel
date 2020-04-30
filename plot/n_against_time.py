import matplotlib.pyplot as plt
import matplotlib.transforms as trs
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
# import numpy as np

from utils.plot import save_fig


def fig_n_against_time(
        data, condition_labels, background=None,
        vline=None,
        ax=None,
        fig_name=None, fig_folder=None,
        y_label=None, colors=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    if background is not None:
        trans = trs.blended_transform_factory(ax.transData,
                                              ax.transAxes)

        ax.fill_between(range(len(background)), 0, 1, where=background == 1,
                        facecolor='whitesmoke',
                        edgecolor='lightgrey',
                        transform=trans,
                        label='Training')
    if vline is not None:
        ax.axvline(vline, color='red', linestyle=':', lw=4, label='Exam')

    if colors is None:
        colors = [f'C{i}' for i in range(len(condition_labels))]

    for i, dt in enumerate(condition_labels):

        ax.plot(data[i], color=colors[i], label=dt)

    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)

    ax.legend()
    # legend_elements = [Line2D([0], [0], color='red', linestyle=':', lw=4,
    #                           label='Exam'),
    #                    # Line2D([0], [0], marker='o', color='w', label='Scatter',
    #                    #        markerfacecolor='g', markersize=15),
    #                    Patch(facecolor='lightgrey', edgecolor='darkgrey',
    #                          label='Training')]
    # handles, labels = ax.get_legend_handles_labels()
    # for le in legend_elements:
    #     handles.append(le)
    # ax.legend(handles=handles)

    if fig_folder is not None and fig_name is not None:
        save_fig(fig_folder=fig_folder, fig_name=fig_name)
    # else:
    #     plt.show()
