import numpy as np
import matplotlib.pyplot as plt


def success(successes):

    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_suplot(111)
    ax.set_ylabel('success')
    ax.set_xlabel('time')
    ax.set_yticks((0, 1))
    ax.scatter(x=np.arange(len(successes)), y=successes, alpha=0.2, color="black")
    plt.show()
