import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_phi(network):
    x = np.arange(network.t_tot_discrete) * network.dt
    y = np.zeros(network.t_tot_discrete)

    for t in range(network.t_tot_discrete):
        network._update_phi(t)
        y[t] = network.phi

    plt.plot(x, y)
    plt.title("Inhibitory oscillations")
    plt.xlabel("Time (cycles)")
    plt.ylabel("$\phi$")
    plt.xlim(min(x), max(x))

    plt.show()


def plot_average_firing_rate(network):
    average_fr = network.average_firing_rate

    x = np.arange(average_fr.shape[1], dtype=float) * \
        network.dt

    fig, ax = plt.subplots()

    for i, y in enumerate(average_fr):
        ax.plot(x, y, linewidth=0.5, alpha=1)
        if i > 1:
            break

    ax.set_xlabel('Time (cycles)')
    ax.set_ylabel('Average firing rate')
    plt.show()


def plot_attractors(network):
    average_fr = network.average_firing_rate

    fig, ax = plt.subplots()
    im = ax.imshow(average_fr, cmap="jet",
                   extent=[
                        0, average_fr.shape[1] * network.dt,
                        average_fr.shape[0] - 0.5, -0.5
                   ])

    ax.set_xlabel('Time (cycles)')
    ax.set_ylabel("Attractor number")

    fig.colorbar(im, ax=ax)

    ax.set_aspect(aspect='auto')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()

    plt.show()
