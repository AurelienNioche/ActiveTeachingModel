import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from plot.generic import save_fig


def plot(p_recall_value,
         success_value,
         questions,
         success_time=None,
         p_recall_time=None,
         fig_name='memory_trace.pdf'):

    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot(111)
    ax.set_ylabel('Success')
    ax.set_xlabel('Time')
    ax.set_yticks((0, 1))

    n_item = p_recall_value.shape[0]
    n_iteration = success_value.shape[0]

    if success_time is None:
        success_time = np.arange(n_iteration)

    if p_recall_time is None:
        p_recall_time = np.arange(n_iteration)

    array_item = np.arange(n_item)

    max_n_item_per_figure = 100
    n_fig = n_item // max_n_item_per_figure +  \
        int(n_item % max_n_item_per_figure != 0)

    item_groups = np.array_split(array_item, n_fig)

    assert 'pdf' in fig_name
    root = fig_name.split('.pdf')[0]

    for idx_fig, item_gp in enumerate(item_groups):

        n_item = len(item_gp)
        fig, axes = plt.subplots(nrows=n_item, figsize=(5, 0.9*n_item))

        for ax_idx, item in enumerate(item_gp):

            color = 'black'  # f'C{i}'
            ax = axes[ax_idx]
            ax.set_ylabel('Recall')
            ax.set_yticks((0, 1))
            ax.set_ylim((-0.1, 1.1))

            ax.scatter(x=success_time[questions == item],
                       y=success_value[questions == item],
                       alpha=0.2,
                       color=color)

            ax.plot(p_recall_time, p_recall_value[item], alpha=0.2,
                    color=color)
            if ax_idx != n_item-1:
                ax.set_xticks([])

        axes[-1].set_xlabel('Time')

        plt.tight_layout()

        fig_name_idx = root + f'_{idx_fig}.pdf'
        save_fig(fig_name_idx)


def summarize(p_recall, fig_name='memory_trace_summarize.pdf',
              p_recall_time=None, font_size=42):

    if p_recall_time is None:
        p_recall_time = np.arange(p_recall.shape[1])

    fig = plt.figure(figsize=(15, 12))
    ax = fig.subplots()

    mean = np.mean(p_recall, axis=0)
    sem = scipy.stats.sem(p_recall, axis=0)

    min_ = np.min(p_recall, axis=0)
    max_ = np.max(p_recall, axis=0)

    ax.plot(p_recall_time, mean, lw=1.5)
    ax.fill_between(
        p_recall_time,
        y1=mean - sem,
        y2=mean + sem,
        alpha=0.2
    )

    ax.plot(p_recall_time, min_, linestyle=':', color='C0')
    ax.plot(p_recall_time, max_, linestyle=':', color='C0')

    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_ylabel('Prob. of recall', fontsize=font_size)

    ax.set_ylim((-0.01, 1.01))

    save_fig(fig_name=fig_name)


def summarize_over_seen(
        p_recall, seen, fig_name='memory_trace_summarize_over_seen.pdf',
        p_recall_time=None, font_size=42):

    n_iteration = p_recall.shape[1]

    if p_recall_time is None:
        p_recall_time = np.arange(n_iteration)

    p_recall[seen == 0] = np.nan

    fig = plt.figure(figsize=(15, 12))
    ax = fig.subplots()

    # for t in range(n_iteration):
    #     print(t)
    #     print('_'*10)
    #     print(p_recall[:, t])
    #     print(seen[:, t])
    #     print()
    #     if t:
    #         a = p_recall[np.isnan(p_recall[:, t]) == 0, t]
    #         print(a)
    #         print(scipy.stats.sem(a, nan_policy='omit'))
    #     print()
    #     print('_' * 10)

    mean = np.nanmean(p_recall, axis=0)
    # sem = scipy.stats.sem(p_recall, axis=0, nan_policy='omit')
    sem = [
        scipy.stats.sem(p_recall[np.isnan(p_recall[:, t]) == 0, t])
        if t > 0 else 0
        for t in range(n_iteration)
    ]

    min_ = np.nanmin(p_recall, axis=0)
    max_ = np.nanmax(p_recall, axis=0)

    ax.plot(p_recall_time, mean, lw=1.5)

    ax.fill_between(
        p_recall_time,
        y1=mean - sem,
        y2=mean + sem,
        alpha=0.2
    )

    ax.plot(p_recall_time, min_, linestyle=':', color='C0')
    ax.plot(p_recall_time, max_, linestyle=':', color='C0')

    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_ylabel('Prob. of recall', fontsize=font_size)

    ax.set_ylim((-0.01, 1.01))

    save_fig(fig_name=fig_name)
