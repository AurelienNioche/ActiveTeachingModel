import numpy as np
import matplotlib.pyplot as plt
import os

dir_path = 'fig'


def success(successes, fig_name='results.pdf'):

    os.makedirs(dir_path, exist_ok=True)

    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot(111)
    ax.set_ylabel('success')
    ax.set_xlabel('time')
    ax.set_yticks((0, 1))
    ax.scatter(x=np.arange(len(successes)), y=successes, alpha=0.2, color="black")
    plt.savefig(fname=f'{dir_path}/{fig_name}')
