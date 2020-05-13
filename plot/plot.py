import matplotlib.pyplot as plt

from . subplot.n_against_time import fig_n_against_time
from . subplot.p_recall import fig_p_item_seen
from . subplot.param_recovery import \
    fig_parameter_recovery, fig_parameter_recovery_heterogeneous

from utils.plot import save_fig


class DataFig:

    def __init__(self, cond_labels, info, exam,
                 training, threshold, param=None, param_labels=None,
                 cond_labels_param_recovery=None):

        self.cond_labels = cond_labels
        self.info = info
        self.exam = exam
        self.training = training
        self.threshold = threshold

        self.cond_labels_param_recovery = cond_labels_param_recovery
        self.param = param
        self.param_labels = param_labels
        self.data = {}

    def add(self, **kwargs):
        for data_type, d in kwargs.items():
            if data_type not in self.data.keys():
                self.data[data_type] = []
            self.data[data_type].append(d)

    def __getitem__(self, key):
        return self.data[key]

    def keys(self):
        return self.data.keys()


def plot_info(ax, info):

    ax.text(0.5, 0.5, info, fontsize=10, wrap=True, ha='center',
            va='center')

    ax.set_axis_off()


def fig_objective(axes, n_learnt,
                  n_seen,
                  training,
                  info,
                  exam,
                  # objective,
                  cond_labels):

    fig_n_against_time(
        data=n_learnt, y_label="N learnt",
        cond_labels=cond_labels,
        ax=axes[0], background=training, vline=exam)

    fig_n_against_time(
        data=n_seen, y_label="N seen",
        cond_labels=cond_labels,
        ax=axes[1], background=training, vline=exam)

    plot_info(ax=axes[2], info=info)


def plot(data, fig_folder, fig_name=''):

    """
    :param fig_name: string
    :param fig_folder: string
    :param data: FigDataSingle
    :return: None
    """

    n_cond = len(data.cond_labels)
    # n_obs = len(data.observation_labels)

    plot_param_recovery = len(data.cond_labels_param_recovery) > 0
    heterogeneous_param = len(data.param.shape) > 1

    if plot_param_recovery:
        if heterogeneous_param:
            len_param_recovery_row = len(data.cond_labels_param_recovery)
            n_row_param_recovery = data.param.shape[-1]
        else:
            len_param_recovery_row = data.param.shape[-1]
            n_row_param_recovery = 1
    else:
        n_row_param_recovery = 0
        len_param_recovery_row = 0

    n_rows = 2 + n_row_param_recovery
    n_cols_per_row_type = {
        "objective": 3,
        "p": n_cond,
        "param_recovery": len_param_recovery_row
    }

    row_types = ["objective", "p", "param_recovery", "param_recovery"]

    n_cols = max(n_cols_per_row_type.values())

    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows,
                             figsize=(6*n_cols, 6*n_rows))

    for i in range(n_rows):
        n = n_cols_per_row_type[row_types[i]]
        if n_cols > n:
            for ax in axes[i, n:]:
                ax.axis('off')

    fig_objective(
        axes=axes[0, :],
        n_learnt=data["n_learnt"],
        n_seen=data["n_seen"],
        training=data.training,
        info=data.info,
        exam=data.exam,
        # objective=data["objective"],
        cond_labels=data.cond_labels)

    fig_p_item_seen(
        axes=axes[1, :],
        p_recall=data["p"],
        background=data.training,
        vline=data.exam,
        hline=data.threshold,
        cond_labels=data.cond_labels)

    if plot_param_recovery:

        if heterogeneous_param:
            fig_parameter_recovery_heterogeneous(
                cond_labels=data.cond_labels_param_recovery,
                param_labels=data.param_labels,
                post_means=data["post_mean"],
                true_param=data.param,
                axes=axes[2:, :]
            )
        else:
            # n axes := n_parameters
            fig_parameter_recovery(
                cond_labels=data.cond_labels_param_recovery,
                param_labels=data.param_labels,
                post_means=data["post_mean"],
                post_sds=data["post_std"],
                true_param=data.param,
                axes=axes[2, :])

    save_fig(fig_name=f"{fig_name}.pdf", fig_folder=fig_folder)