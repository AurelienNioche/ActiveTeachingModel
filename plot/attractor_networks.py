import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_currents(network):

    data = network.currents

    fig, ax = plt.subplots()
    im = ax.imshow(data)
    ax.set_aspect("auto")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")

    ax.set_title("Network currents history")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Iteration")

    plt.tight_layout()
    plt.show()


def plot_weights(network):

    fig, ax = plt.subplots()
    im = ax.contourf(network.weights)
    ax.set_aspect("auto")

    ax.set_title("Weights matrix")
    ax.set_xlabel("Neuron id")
    ax.set_ylabel("Neuron id")

    plt.tight_layout()

    fig.colorbar(im, ax=ax)
    plt.show()
