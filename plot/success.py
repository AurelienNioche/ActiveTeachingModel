import numpy as np
import matplotlib.pyplot as plt

from plot.generic import save_fig


def scatter(successes, fig_name='backup.pdf'):

    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot(111)
    ax.set_ylabel('Success')
    ax.set_xlabel('Time')
    ax.set_yticks((0, 1))
    ax.scatter(x=np.arange(len(successes)), y=successes, alpha=0.2, color="black")

    save_fig(fig_name)


def curve(successes, fig_name=None, font_size=12, line_width=1,
          label_size=8, ax=None, color='C0'):

    y = []
    for i in range(1, len(successes)):
        y.append(np.mean(successes[:i]))

    n_iteration = len(y)

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

    ax.plot(y, color=color, linewidth=line_width)

    # Both axis
    ax.tick_params(axis="both", labelsize=label_size)

    # x-axis
    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_xlim(1, n_iteration)
    ax.set_xticks((1, int(n_iteration/2), n_iteration))

    # y-axis
    ax.set_ylabel('Success rate', fontsize=font_size)
    ax.set_ylim((-0.01, 1.01))
    ax.set_yticks((0, 0.5, 1))

    if fig_name is not None:
        save_fig(fig_name)


def multi_curve(questions, replies, fig_name, font_size=42, line_width=3,
                label_size=22, max_lines=None):

    n_items = len(np.unique(questions))
    t_max = len(questions)

    if max_lines is None:
        cumulative_success = np.zeros(n_items)
        cumulative_seen = np.zeros(n_items)
        data = np.zeros((n_items, t_max))

    else:
        cumulative_success = np.zeros(max_lines)
        cumulative_seen = np.zeros(max_lines)
        data = np.zeros((max_lines, t_max))

    for t, (q, r) in enumerate(zip(questions, replies)):

        if max_lines is None or q < max_lines:
            cumulative_seen[q] += 1
            cumulative_success[q] += int(q == r)

        sup0 = cumulative_seen > 0
        data[sup0, t] = cumulative_success[sup0] / cumulative_seen[sup0]

    if max_lines is not None:

        data = data[:max_lines]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_ylabel('Success rate', fontsize=font_size)
    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_yticks((0, 1))
    ax.set_ylim((0, 1))
    ax.tick_params(axis="both", labelsize=label_size)
    ax.plot(data.T, linewidth=line_width, alpha=0.5)

    plt.tight_layout()

    save_fig(fig_name)
