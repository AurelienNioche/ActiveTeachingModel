import string

from plot import \
    fig_parameter_recovery, \
    fig_n_against_time, fig_p_item_seen

import matplotlib.pyplot as plt

from utils.plot import save_fig, add_letter


class DataFigSingle:

    def __init__(self, param=None, param_labels=None):

        self.n_learnt = []
        self.objective = []
        self.n_seen = []
        self.labels = []
        self.p_item = []
        self.post_mean = []
        self.post_std = []

        self.param = param
        self.param_labels = param_labels

    def add(self, n_learnt, n_seen, p_item, label,
            post_mean=None, post_std=None, objective=None):

        self.n_learnt.append(n_learnt)
        self.n_seen.append(n_seen)
        self.p_item.append(p_item)
        self.labels.append(label)
        self.post_mean.append(post_mean)
        self.post_std.append(post_std)
        self.objective.append(objective)


def fig_single(data, fig_folder, time_scale=(60*60*24)/2, fig_name=''):

    """
    :param fig_name: string
    :param time_scale: float
    :param fig_folder: string
    :param data: FigDataSingle
    :return: None
    """

    data_for_objective = data.objective[0] is not None
    data_for_post = None not in data.post_mean

    n_cond = len(data.labels)

    n_rows = 2 + (1 if data_for_post else 0)
    n_cols_per_row = {
        0: 2 + (1 if data_for_objective else 0),
        1: n_cond,
        2: len(data.param_labels) if data_for_post else 0
    }

    n_cols = max(n_cols_per_row.values())

    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(12, 9))

    ax_n_learnt = axes[0, 0]
    ax_n_seen = axes[0, 1]
    ax_objective = axes[0, 2] if data_for_objective else None

    for i in range(n_rows):
        if n_cols > n_cols_per_row[i]:
            for ax in axes[i, n_cols_per_row[i]:]:
                ax.axis('off')

    # n axes := n_conditions
    ax_p_item = axes[1, :]

    where_to_put_letter = []  # [ax_objective, ax_n_seen, ax_p_item[0], ]

    for i, ax in enumerate(where_to_put_letter):
        add_letter(ax=ax, letter=string.ascii_uppercase[i])

    fig_n_against_time(
        data=data.n_learnt, y_label="N learnt",
        condition_labels=data.labels,
        ax=ax_n_learnt)

    if data_for_objective:

        fig_n_against_time(
            data=data.objective, y_label="Objective",
            condition_labels=data.labels,
            ax=ax_objective)

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

    save_fig(fig_name=f"{fig_name}.pdf", fig_folder=fig_folder)
