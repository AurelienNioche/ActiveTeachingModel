import matplotlib.pyplot as plt
import numpy as np

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

    fig, axes = plt.subplots(nrows=n_item, figsize=(5, 0.9*n_item))

    for i in range(n_item):

        color = 'black'  # f'C{i}'
        ax = axes[i]
        ax.set_ylabel('Recall')
        ax.set_yticks((0, 1))
        ax.set_ylim((-0.1, 1.1))

        ax.scatter(x=success_time[questions == i],
                   y=success_value[questions == i],
                   alpha=0.2,
                   color=color)

        ax.plot(p_recall_time, p_recall_value[i], alpha=0.2,
                color=color)
        if i != n_item-1:
            ax.set_xticks([])

    axes[-1].set_xlabel('Time')

    plt.tight_layout()
    save_fig(fig_name)
