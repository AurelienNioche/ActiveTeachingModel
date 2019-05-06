import numpy as np
import matplotlib.pyplot as plt
import os

dir_path = 'fig'


def save_fig(fig_name):

    os.makedirs(dir_path, exist_ok=True)

    file_name = f'{dir_path}/{fig_name}'
    plt.savefig(fname=f'{dir_path}/{fig_name}')

    print(f'Figure "{file_name}" created.')
    plt.close()


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

    os.makedirs(dir_path, exist_ok=True)

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
