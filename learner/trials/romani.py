import pickle
import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
import itertools as it
from matplotlib.ticker import MaxNLocator

np.seterr(all='raise')


class Network:

    def __init__(self,
                 L=16,
                 N=3000,
                 T=0.015,
                 T_th=45,
                 T_j0=25,
                 J_0_min=0.7,
                 J_0_max=1.2,
                 t_max=1000,
                 f=0.1,
                 phase_shift=0.0,
                 first_p=0,
                 seed=123):

        np.random.seed(seed)

        self.L = L
        self.N = N
        self.T = T
        self.T_th = T_th
        self.T_j0 = T_j0
        self.J_0_min = J_0_min
        self.J_0_max = J_0_max
        self.f = f
        self.t_max = t_max

        self.D_th = 1.9 * T
        self.phase_shift = phase_shift
        self.first_p = first_p

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

        self.idx_neuron = np.arange(self.N)

        self.n_factor = 1 / \
            (
                self.N * self.f * (1-self.f)
            )

    @staticmethod
    def sinusoid(min_, max_, period, t, phase_shift):

        amplitude = (max_ - min_) / 2
        frequency = (1 / period)  # * self.pr.dt
        # phase = np.random.choice()
        shift = min_ + amplitude  # Moves the wave in the y-axis

        return \
            amplitude \
            * np.sin(2 * np.pi * (t + phase_shift) * frequency) \
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
            period=self.T_j0,
            phase_shift=self.phase_shift*self.T_j0,
            t=t
        )

    def _build_connections(self):

        print('Building the connections...')

        # coordinates = list(it.product(range(self.N), repeat=2))

        for i in tqdm(range(self.N)):
            for j in range(self.N):
                # if i == j:
                #     continue
                sum_ = 0
                for mu in range(self.L):
                    sum_ += \
                        (self.J[mu, i] - self.f) \
                        * (self.J[mu, j] - self.f)

                self.J[i, j] = sum_ * self.n_factor

    def _update_threshold(self):

        for i in range(self.N):
            self.th[i] = self.th[i] - \
                (self.th[i] - self.first_th[i]
                 - self.D_th*self.previous_V[i]) \
                / self.T_th

    def _update_i(self, i):

        # print(f'updating neuron {i}...')
        sum_ = 0
        for j in range(self.N):
            sum_ += \
                self.J[i, j] * self.previous_V[j]

        second_sum_ = np.sum(self.previous_V)

        multiplier = self.J_0 / (self.N * self.f)

        inside_parenthesis = \
            sum_ - multiplier * second_sum_ - self.th[i]

        return self.heaviside(inside_parenthesis)

    def _update_activation(self):

        self.previous_V = self.V.copy()

        self.V[:] = mp.Pool().map(self._update_i, range(self.N))

        # for i in range(self.N):
        #     self.V[i] = self._update_i(i)

    def _present_pattern(self):

        self.V[:] = self.xi[self.first_p, :]

    def _save_population_activity(self, t):

        for mu in range(self.L):
            sum_ = 0
            for i in range(self.N):
                sum_ += \
                    (self.xi[mu, i] - self.f) * self.V[i]

            self.population_activity[mu, t] = sum_ * self.n_factor

        # self.population_activity[:, t] *= self.n_factor

    def simulate(self):

        self._build_connections()
        self._present_pattern()
        # self._save_population_activity(0)

        print(f"Simulating for {self.t_max} time steps...\n")
        for t in tqdm(range(self.t_max)):

            self._update_J_0(t)
            self._update_activation()
            self._update_threshold()

            self._save_population_activity(t)


def plot_attractors(activity):

    fig, ax = plt.subplots()
    im = ax.imshow(activity, aspect='auto',
                   cmap='jet')
    fig.colorbar(im, ax=ax)

    ax.set_xlabel('Time')
    ax.set_ylabel("Memories")

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

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
            phase_shift=0.75,
        )

    plot(history)


def main(force=False):

    N = 3000
    t_max = 1000
    L = 16
    phase_shift = 0.75

    bkp_file = os.path.join(
        'bkp',
        f'romani_N{N}_tmax{t_max}_L{L}_phase_shift{phase_shift}.p')

    os.makedirs(os.path.dirname(bkp_file), exist_ok=True)

    if not os.path.exists(bkp_file) or force:

        network = Network(
            N=N,
            t_max=t_max,
            L=L,
            phase_shift=0.75,
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
