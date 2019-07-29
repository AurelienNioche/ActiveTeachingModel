import numpy as np
import matplotlib.pyplot as plt


def phi(phi_min=0.7, phi_max=1.06, t=0, tau_0=1, dt=0.001):

    amplitude = (phi_max - phi_min) / 2
    frequency = (1 / tau_0) * dt
    # phase = np.random.choice()
    shift = phi_min + amplitude  # Moves the wave in the y-axis

    _phi = amplitude * np.sin(2 * np.pi * t * frequency) + shift

    assert phi_min <= _phi <= phi_max

    return _phi


def plot_phi(phi_history):

    time = np.arange(0, phi_history.size, 1)

    plt.plot(time, phi_history)
    plt.title("Inhibitory oscillations")
    plt.xlabel("Time")
    plt.ylabel("Phi")

    plt.show()

    print(np.amax(phi_history))
    print(np.amin(phi_history))


def main():

    t_max = 1000

    phi_history = np.zeros(t_max)
    for t in range(t_max):
        phi_history[t] = phi(t=t)

    plot_phi(np.asarray(phi_history))


if __name__ == "__main__":
    main()
