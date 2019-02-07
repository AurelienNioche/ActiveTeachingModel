import numpy as np
import matplotlib.pyplot as plt


def success(successes):

    plt.scatter(x=np.arange(len(successes)), y=successes, alpha=0.2, color="black")
    plt.show()
