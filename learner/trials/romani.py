import pickle
import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

np.seterr(all='raise')


class Network:

    version = 0.1
    bounds = ('d', 0.001, 1.0), \
             ('tau', -1, 1), \
             ('s', 0.001, 1)  # TODO change

    def __init__(self,
                 L=16,
                 N=3000,
                 T=0.015,
                 T_th=45,
                 T_jo=25,
                 J_0_min=0.7,
                 J_0_max=1.2,
                 t_max=1000,
                 f=0.1,
                 phase_shift=0.5,
                 seed=123):

        np.random.seed(seed)

        self.L = L
        self.N = N
        self.T = T
        self.T_th = T_th
        self.T_jo = T_jo
        self.J_0_min = J_0_min
        self.J_0_max = J_0_max
        self.f = f
        self.t_max = t_max

        self.D_th = 1.9 * T
        self.phase_shift = phase_shift

        self.J = np.zeros((self.N, self.N))
        self.V = np.zeros(self.N)

        # Phi-like parameter
        self.J_0 = None

        self.xi = \
            np.random.choice([0, 1], p=[1 - self.f, self.f],
                             size=(self.L, self.N))

        self.th = \
            np.random.uniform(-self.T, self.T, size=self.N)

        self.first_th = self.th.copy()

        self.previous_V = np.zeros(self.N)

        self.population_activity = \
            np.zeros((self.L, self.t_max))

    @staticmethod
    def sinusoid(min_, max_, period, t, phase_shift):

        amplitude = (max_ - min_) / 2
        frequency = (1 / period)  # * self.pr.dt
        # phase = np.random.choice()
        shift = min_ + amplitude  # Moves the wave in the y-axis

        return amplitude \
               * np.sin(2 * np.pi * (t + phase_shift*period) * frequency) \
               + shift

    @staticmethod
    def heaviside(x):

        return int(x >= 0)

    def _update_J_0(self, t):
        """
        J0 is a sinusoid function related to neuron inhibition.
        It follows the general sine wave function:
            f(x) = amplitude * (2 * pi * frequency * time)
        :param t: int discrete time step.
        """
        self.J_0 = self.sinusoid(
            min_=self.J_0_min,
            max_=self.J_0_max,
            period=self.T_jo,
            phase_shift=self.phase_shift,
            t=t
        )

    def _build_connections(self):
        print('Building the connections...')

        multiplier = 1 / \
                     (
                        self.N *  self.f * (1-self.f)
                     )

        for i in tqdm(range(self.J.shape[0])):
            for j in range(self.J.shape[1]):
                sum_ = 0
                for mu in range(self.L):
                    sum_ += \
                        (self.J[mu, i] - self.f) \
                        * (self.J[mu, j] - self.f)

                self.J[i, j] = \
                     multiplier * sum_
        # print("Done!\n")

    def _update_threshold(self):

        self.th[:] -= \
            (self.th[:] - self.first_th[:] - self.D_th*self.previous_V[:]) \
            / self.T_th

    def _update_activation(self):

        self.previous_V = self.V.copy()

        # order = np.arange(self.N)
        # np.random.shuffle(order)
        # print("order", order)
        # print("\n")
        # print("-" * 5)

        for i in range(self.N):

            # print(f'updating neuron {i}...')
            sum_ = 0
            for j in range(self.N):
                sum_ += \
                    self.J[i, j] * self.previous_V[j]

            second_sum_ = np.sum(self.previous_V)

            multiplier = self.J_0 / (self.N*self.f)

            inside_parenthesis = \
                sum_ - multiplier * second_sum_ - self.th[i]

            # print("-" * 5)

            self.V[i] = self.heaviside(inside_parenthesis)

    def _present_pattern(self):

        chosen_pattern_number = np.random.choice(np.arange(self.L))
        print(f"Chosen pattern number {chosen_pattern_number} of {self.L}")

        self.V[:] = self.xi[chosen_pattern_number, :]
        # print('\nactivation\n', self.activation)

    def _save_population_activity(self, t):

        multiplier = 1 / \
                     (
                             self.N * self.f * (1 - self.f)
                     )

        for mu in range(self.L):
            sum_ = 0
            for i in range(self.N):
                sum_ += \
                    (self.xi[mu, i] - self.f) * self.V[i]

            self.population_activity[mu, t] = \
                sum_

            # try:
            #     self.population_activity[mu, t] = \
            #         np.mean(self.V[self.xi[mu, :] == 1])
            # except FloatingPointError:
            #     self.population_activity[mu, t] = 0

    def simulate(self):

        self._build_connections()

        self._present_pattern()

        self._save_population_activity(0)

        print(f"Simulating for {self.t_max} time steps...\n")
        for t in tqdm(range(1, self.t_max)):
            # print("*" * 10)
            # print("T", t)
            # print("*" * 10)
            self._update_J_0(t)
            self._update_activation()
            self._update_threshold()

            self._save_population_activity(t)

            # break


# def plot_phi(network):
#     data = network.phi_history
#     time = np.arange(0, network.phi_history.size, 1)
#
#     plt.plot(time, data)
#     plt.title("Inhibitory oscillations")
#     plt.xlabel("Time")
#     plt.ylabel("Phi")
#
#     plt.show()
#
#     print(np.amax(network.phi_history))
#     print(np.amin(network.phi_history))


def plot_weights(weights, time):

    fig, ax = plt.subplots()
    im = ax.imshow(weights)

    plt.title(f"Weights matrix (t = {time})")
    plt.xlabel("Weights")
    plt.ylabel("Weights")

    fig.tight_layout()

    plt.show()


# def plot_average_fr(average_fr, x_scale=1000):
#
#     x = np.arange(average_fr.shape[1], dtype=float) / x_scale
#
#     fig, ax = plt.subplots()
#
#     for i, y in enumerate(average_fr):
#         ax.plot(x, y, linewidth=0.5, alpha=0.2)
#         if i > 1:
#             break
#
#     ax.set_xlabel('Time (cycles)')
#     ax.set_ylabel('Average firing rate')
#     plt.show()


def plot_attractors(activity, x_scale=1000):

    fig, ax = plt.subplots()  #(figsize=(10, 100))
    im = ax.imshow(activity, aspect='auto')
    fig.colorbar(im, ax=ax)

    ax.set_xlabel('Time')
    ax.set_ylabel("Memories")

    # ax.set_aspect('equal', 'datalim')

    fig.tight_layout()

    plt.show()


def _test_sinusoid():

    def plot(h):

        time = np.arange(0, h.size, 1)

        plt.plot(time, h)
        plt.title("Inhibitory oscillations")
        plt.xlabel("Time")
        plt.ylabel("y")

        plt.show()

    t_max = 100

    history = np.zeros(t_max)
    for t in range(t_max):
        history[t] = Network.sinusoid(
            t=t,
            min_=0.7,
            max_=1.2,
            period=25,
            phase_shift=25/2,
        )

    plot(history)


def main(force=False):

    N = 3000
    t_max = 300

    bkp_file = f'bkp/romani_N{N}_tmax{t_max}.p'

    os.makedirs(os.path.dirname(bkp_file), exist_ok=True)

    if not os.path.exists(bkp_file) or force:

        network = Network(
            N=N,
            t_max=t_max,
        )
        network.simulate()

        pickle.dump(network, open(bkp_file, 'wb'))
    else:
        print('Loading from pickle file...')
        network = pickle.load(open(bkp_file, 'rb'))

    plot_attractors(network.population_activity)


if __name__ == "__main__":
    # _test_sinusoid()
    main(force=True)
