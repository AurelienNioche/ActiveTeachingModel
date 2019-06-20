from matplotlib import pyplot as plt

from plot.generic import save_fig


def bar(counts, fig_name, font_size=42, label_size=22):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_ylabel('N', fontsize=font_size)
    ax.set_xlabel('Item', fontsize=font_size)
    # ax.set_yticks((0, 1))
    # ax.set_ylim((0, 1))
    ax.tick_params(axis="both", labelsize=label_size)
    ax.bar(range(len(counts)), counts)

    plt.tight_layout()

    save_fig(fig_name)


def scatter(fig_name, font_size=42, label_size=22):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_ylabel('N', fontsize=font_size)
    ax.set_xlabel('Item', fontsize=font_size)
    # ax.set_yticks((0, 1))
    # ax.set_ylim((0, 1))
    ax.tick_params(axis="both", labelsize=label_size)

    ax.scatter(range(len(counts)), counts)

    plt.tight_layout()

    save_fig(fig_name)