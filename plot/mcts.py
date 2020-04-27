from plot import fig_n_against_time, fig_p_item_seen

import matplotlib.pyplot as plt

from utils.plot import save_fig


class DataFig:

    def __init__(self, observation_labels, condition_labels):

        self.condition_labels = condition_labels
        self.observation_labels = observation_labels
        self.data = {k: {} for k in observation_labels}

    def add(self, **kwargs):
        for data_type, d in kwargs.items():
            for k_obs, v in d.items():
                if data_type not in self.data[k_obs].keys():
                    self.data[k_obs][data_type] = []
                self.data[k_obs][data_type].append(v)

    def __getitem__(self, key):
        return self.data[key]


def fig_objective(axes, n_learnt,
                  n_seen,
                  objective,
                  condition_labels):

    fig_n_against_time(
        data=n_learnt, y_label="N learnt",
        condition_labels=condition_labels,
        ax=axes[0])

    fig_n_against_time(
        data=objective, y_label="Objective",
        condition_labels=condition_labels,
        ax=axes[1])

    fig_n_against_time(
        data=n_seen, y_label="N seen",
        condition_labels=condition_labels,
        ax=axes[2])


def make_fig(data, fig_folder, fig_name=''):

    """
    :param fig_name: string
    :param fig_folder: string
    :param data: FigDataSingle
    :return: None
    """

    n_cond = len(data.condition_labels)
    n_obs = len(data.observation_labels)

    n_rows = 2 * n_obs
    n_cols_per_row_type = {
        "objective": 3,
        "p": n_cond
    }

    row_types = ["objective", "p", "objective", "p"]

    n_cols = max(n_cols_per_row_type.values())

    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(12, 9))

    for i in range(n_rows):
        n = n_cols_per_row_type[row_types[i]]
        if n_cols > n:
            for ax in axes[i, n:]:
                ax.axis('off')

    i = 0
    for k_obs in data.observation_labels:

        fig_objective(
            axes=axes[i, :],
            n_learnt=data[k_obs]["n_learnt"],
            n_seen=data[k_obs]["n_seen"],
            objective=data[k_obs]["objective"],
            condition_labels=data.condition_labels)

        fig_p_item_seen(
            axes=axes[i + 1, :],
            p_recall=data[k_obs]["p"],
            condition_labels=data.condition_labels)

        i += 2

    save_fig(fig_name=f"{fig_name}.pdf", fig_folder=fig_folder)
