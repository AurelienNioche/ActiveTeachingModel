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


def curve(successes, fig_name='backup.pdf', font_size=42, line_width=3,
          label_size=22):

    y = []
    for i in range(1, len(successes)):
        y.append(np.mean(successes[:i]))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_ylabel('Success rate', fontsize=font_size)
    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_yticks((0, 1))
    ax.set_ylim((0, 1))
    ax.tick_params(axis="both", labelsize=label_size)
    ax.plot(y, color="black", linewidth=line_width)

    plt.tight_layout()

    save_fig(fig_name)


def multi_curve(questions, replies, fig_name, font_size=42, line_width=3,
                label_size=22, max_lines=None):

    n_items = len(np.unique(questions))
    t_max = len(questions)

    cumulative_success = np.zeros(n_items)
    cumulative_seen = np.zeros(n_items)
    data = np.zeros((n_items, t_max))

    for t, (q, r) in enumerate(zip(questions, replies)):
        cumulative_seen[q] += 1
        cumulative_success[q] += int(q == r)
        data[q, t] = cumulative_success[q] / cumulative_seen[q]

    if max_lines is not None:

        data = data[:max_lines]

    print(data.shape)

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

