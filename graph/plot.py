import numpy as np
import matplotlib.pyplot as plt
import os

dir_path = 'fig'


def success_scatter_plot(successes, fig_name='backup.pdf'):

    os.makedirs(dir_path, exist_ok=True)

    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot(111)
    ax.set_ylabel('Success')
    ax.set_xlabel('Time')
    ax.set_yticks((0, 1))
    ax.scatter(x=np.arange(len(successes)), y=successes, alpha=0.2, color="black")
    plt.savefig(fname=f'{dir_path}/{fig_name}')
    plt.close()


def success_curve(successes, fig_name='backup.pdf'):

    y = []
    for i in range(1, len(successes)):
        y.append(np.mean(successes[:i]))

    print(y)

    os.makedirs(dir_path, exist_ok=True)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_ylabel('Success rate')
    ax.set_xlabel('Time')
    ax.set_yticks((0, 1))
    ax.set_ylim((0, 1))
    ax.plot(y, color="black")
    plt.savefig(fname=f'{dir_path}/{fig_name}')
    plt.close()
