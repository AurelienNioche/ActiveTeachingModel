import matplotlib.pyplot as plt
import numpy as np

from plot.generic import save_fig


def plot(p_recall_value,
         p_recall_time,
         success_time,
         success_value,
         questions,
         fig_name='memory_trace.pdf'):

    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot(111)
    ax.set_ylabel('Success')
    ax.set_xlabel('Time')
    ax.set_yticks((0, 1))

    n_item = p_recall_value.shape[0]

    assert n_item <= 10, success_time.shape

    for i in range(n_item):

        color = f'C{i}'

        ax.scatter(x=success_time[questions == i],
                   y=success_value[questions == i],
                   alpha=0.2,
                   color=color)

        ax.plot(p_recall_time, p_recall_value[i], alpha=0.2,
                color=color)

    save_fig(fig_name)
