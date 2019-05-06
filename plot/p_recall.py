import matplotlib.pyplot as plt

from plot.generic import save_fig


def curve(p_recall, fig_name='p_recall.pdf', font_size=42, line_width=3,
          label_size=22, alpha=0.5):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_ylabel('Probability of recall', fontsize=font_size)
    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_yticks((0, 1))
    ax.set_ylim((0, 1))
    ax.tick_params(axis="both", labelsize=label_size)

    ax.plot(p_recall, linewidth=line_width, alpha=alpha)

    save_fig(fig_name)