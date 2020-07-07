import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from . subplot.n_against_time import fig_n_against_time
from . subplot.p_recall import fig_p_item_seen
from . subplot.param_recovery import fig_parameter_recovery

from utils.plot import save_fig


class DataFig:

    def __init__(self, xlims, cond_labels, info, exam,
                 time_per_iter,
                 training, threshold, param=None, param_labels=None,
                 cond_labels_param_recovery=None):

        self.xlims = xlims
        self.time_per_iter = time_per_iter
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


def plot(data, fig_folder, fig_name=''):

    """
    :param fig_name: string
    :param fig_folder: string
    :param data: FigDataSingle
    :return: None
    """

    timestamps = data['timestamps']
    n_learnt = data["n_learnt"]
    n_seen = data["n_seen"]
    p_recall = data["p"]
    xlims = data.xlims
    time_per_iter = data.time_per_iter
    cond_labels = data.cond_labels
    cond_labels_param_recovery = data.cond_labels_param_recovery
    training = data.training
    exam = data.exam
    info = data.info
    threshold = data.threshold

    n_cond = len(data.cond_labels)

    plot_param_recovery = len(data.cond_labels_param_recovery) > 0
    heterogeneous_param = len(data.param.shape) > 1

    if plot_param_recovery:
        timestamps_recovery = data["timestamps_recovery"]
        inferred_param = data["inferred_param"]
        param_labels = data.param_labels
        true_param = data.param
        if heterogeneous_param:
            len_param_recovery_row = len(data.cond_labels_param_recovery)
            n_row_param_recovery = data.param.shape[-1]
        else:
            len_param_recovery_row = data.param.shape[-1]
            n_row_param_recovery = 1
    else:
        timestamps_recovery = None
        inferred_param = None
        param_labels = None
        true_param = None
        n_row_param_recovery = 0
        len_param_recovery_row = 0

    n_rows = 2 + n_row_param_recovery

    n_cols_per_row = [3, n_cond, len_param_recovery_row]
    n_cols = max(n_cols_per_row)

    fig = plt.figure(figsize=(6*n_cols, 6*n_rows))
    gs = gridspec.GridSpec(ncols=n_cols, nrows=n_rows, figure=fig)

    colors = [f'C{i}' for i in range(len(cond_labels))]

    fig_n_against_time(
        gs=gs[0, 0],
        fig=fig,
        time_per_iter=time_per_iter,
        timestamps=timestamps,
        xlims=xlims,
        data=n_learnt,
        ylabel="N learnt",
        cond_labels=cond_labels,
        background=training,
        colors=colors,
        vline=exam)

    fig_n_against_time(
        fig=fig,
        gs=gs[0, 1],
        xlims=xlims,
        time_per_iter=time_per_iter,
        timestamps=timestamps,
        data=n_seen, ylabel="N seen",
        cond_labels=cond_labels,
        background=training,
        colors=colors,
        vline=exam)

    plot_info(ax=fig.add_subplot(gs[0, 2]), info=info)

    for i, dt in enumerate(cond_labels):

        fig_p_item_seen(
            gs=gs[1, i],
            fig=fig,
            xlims=xlims,
            label=dt,
            p_recall=p_recall[i],
            time_per_iter=time_per_iter,
            background=training,
            vline=exam,
            hline=threshold,
            color=colors[i])

    if plot_param_recovery:

        if not heterogeneous_param:
            n_param = len(data.param_labels)
            for i in range(n_param):
                pr = param_labels[i]
                fig_parameter_recovery(
                    fig=fig,
                    gs=gs[2, i],
                    xlims=xlims,
                    pr=pr,
                    param_labels=param_labels,
                    cond_labels=cond_labels_param_recovery,
                    timestamps=timestamps_recovery,
                    inferred_param=inferred_param,
                    true_param=true_param,
                    colors=colors,
                    background=training,
                    time_per_iter=time_per_iter)
    #
    #     if heterogeneous_param:
    #         fig_parameter_recovery_heterogeneous(
    #             cond_labels=data.cond_labels_param_recovery,
    #             param_labels=data.param_labels,
    #             post_means=data["post_mean"],
    #             true_param=data.param,
    #             axes=axes[2:, :]
    #         )
    #     else:
    #         # n axes := n_parameters
    #         fig_parameter_recovery(
    #             cond_labels=data.cond_labels_param_recovery,
    #             param_labels=data.param_labels,
    #             post_means=data["post_mean"],
    #             post_sds=data["post_std"],
    #             true_param=data.param,
    #             axes=axes[2, :])

    save_fig(fig_name=f"{fig_name}.pdf", fig_folder=fig_folder)