import os

import numpy as np
import string


from utils.string import dic2string

from plot import \
    fig_parameter_recovery, \
    fig_p_recall, fig_n_seen, fig_p_item_seen

from model.constants import \
    POST_MEAN, POST_SD, P_SEEN, N_SEEN, N_LEARNT, P_ITEM

import matplotlib.pyplot as plt

from utils.plot import save_fig


EPS = np.finfo(np.float).eps
FIG_FOLDER = os.path.join("fig", os.path.basename(__file__).split(".")[0])
os.makedirs(FIG_FOLDER, exist_ok=True)


class DataSingle:

    def __init__(self):

        self._learnt = []
        self._n_seen = []
        self._label = []
        self._p_item = []

    def add(self, learnt, n_seen, p_item, label):

        self._learnt.append(learnt)
        self._n_seen.append(n_seen)
        self._p_item.append(p_item)
        self._label.append(label)

    @property
    def learnt(self):
        return

    def _dic_data(self, attr):
        return {
            k: v for (k, v) in zip(self._label, attr)
        }


def plot_single(data_learnt,
                data_n_seen,
                data_p_item,
                condition_labels, ):

    fig_ext = \
        "_" \
        f"{learner_model.__name__}_" \
        f"{dic2string(param)}_" \
        f"{n_session}session" \
        f".pdf"
    #
    # # ax.text(-0.2, 1.2, string.ascii_uppercase[row],
    # #         transform=ax.transAxes,
    # #         size=20, weight='bold')
    #
    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(12, 9))

    ax = axes[0, 0]
    fig_n_seen(
        data=data[N_LEARNT], y_label="N learnt",
        condition_labels=condition_labels,
        ax=ax)

    ax.text(-0.1, -0.1, string.ascii_uppercase[0],
                    transform=ax.transAxes, size=20, weight='bold')

    ax = axes[0, 1]
    fig_n_seen(
        data=data[N_SEEN], y_label="N seen",
        condition_labels=condition_labels,
        ax=ax)

    ax.text(-0.1, -0.1, string.ascii_uppercase[1],
            transform=ax.transAxes, size=20, weight='bold')


    # # n axes := n_parameters
    # fig_parameter_recovery(condition_labels=condition_labels,
    #                        param_labels=param_labels,
    #                        post_means=data[POST_MEAN], post_sds=data[POST_SD],
    #                        true_param=param,
    #                        axes=axes[1, :])

    ax = axes[1, 0]
    ax.text(-0.1, -0.1, string.ascii_uppercase[2],
                    transform=ax.transAxes, size=20, weight='bold')

    # n axes := n_conditions
    fig_p_item_seen(
        p_recall=data[P_ITEM], condition_labels=condition_labels,
        axes=axes[2, :]
        )

    ax = axes[2, 0]
    ax.text(-0.1, -0.1, string.ascii_uppercase[3],
                    transform=ax.transAxes, size=20, weight='bold')
    #
    # fig_p_recall(data=data[P_SEEN], condition_labels=condition_labels,
    #              ax=axes[5])
    #
    save_fig(fig_name=f"single{fig_ext}", fig_folder=FIG_FOLDER)