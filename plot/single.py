import os

import numpy as np
import string


from utils.string import dic2string

from plot import \
    fig_parameter_recovery, \
    fig_p_recall, fig_n_against_time, fig_p_item_seen

from model.constants import \
    POST_MEAN, POST_SD, P_SEEN, N_SEEN, N_LEARNT, P_ITEM

import matplotlib.pyplot as plt

from utils.plot import save_fig


# EPS = np.finfo(np.float).eps


class DataFigSingle:

    def __init__(self, learner_model, n_session, param, param_labels, ):

        self.learner_model = learner_model
        self.param = param
        self.param_labels = param_labels
        self.n_session = n_session

        self.objective = []
        self.n_seen = []
        self.labels = []
        self.p_item = []
        self.post_mean = []
        self.post_std = []

    def add(self, objective, n_seen, p_item, label,
            post_mean=None, post_std=None):

        self.objective.append(objective)
        self.n_seen.append(n_seen)
        self.p_item.append(p_item)
        self.labels.append(label)
        self.post_mean.append(post_mean)
        self.post_std.append(post_std)


def add_letter(ax, letter):
    ax.text(-0.1, -0.1, letter,
            transform=ax.transAxes, size=20, weight='bold')


def fig_single(data, fig_folder, time_scale=(60*60*24)/2, ext=''):

    """
    :param time_scale: float
    :param fig_folder: string
    :param data: FigDataSingle
    :return: None
    """
    param_string = \
        '_'.join(f'{k}={v:.2f}'
                 for (k, v) in zip(data.param_labels, data.param))

    fig_ext = \
        f"{data.learner_model.__name__}_" \
        f"{param_string}_" \
        f"n_session={data.n_session}{ext}"

    n_rows = 2 + int(None not in data.post_mean)
    n_cols = len(data.labels)

    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(12, 9))

    ax_objective = axes[0, 0]
    ax_n_seen = axes[0, 1]
    if len(data.labels) > 2:
        for ax in axes[0, 2:]:
            ax.axis('off')

    # n axes := n_conditions
    ax_p_item = axes[1, :]

    where_to_put_letter = [ax_objective, ax_n_seen, ax_p_item[0], ]

    fig_n_against_time(
        data=data.objective, y_label="Objective",
        condition_labels=data.labels,
        ax=ax_objective)

    for i, ax in enumerate(where_to_put_letter):
        add_letter(ax=ax, letter=string.ascii_uppercase[i])

    fig_n_against_time(
        data=data.n_seen, y_label="N seen",
        condition_labels=data.labels,
        ax=ax_n_seen)

    # n axes := n_conditions
    fig_p_item_seen(
        p_recall=data.p_item, condition_labels=data.labels,
        axes=ax_p_item,
        time_scale=time_scale)

    if None not in data.post_mean:

        # n axes := n_parameters
        ax_param_rec = axes[2, :]
        where_to_put_letter.append(ax_param_rec[0])

        # n axes := n_parameters
        fig_parameter_recovery(condition_labels=data.labels,
                               param_labels=data.param_labels,
                               post_means=data.post_mean,
                               post_sds=data.post_std,
                               true_param=data.param,
                               axes=ax_param_rec)

    save_fig(fig_name=f"single_{fig_ext}.pdf", fig_folder=fig_folder)

    # fig_p_recall(data=data[P_SEEN], condition_labels=condition_labels,
    #              ax=axes[5])
