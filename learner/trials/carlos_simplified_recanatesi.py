import os
import pickle

import numpy as np
from tqdm import tqdm

import plot.attractor_networks
# from learner.carlos_recanatesi import Network

import matplotlib.pyplot as plt

np.seterr(all='raise')


class SimplifiedNetwork:
    """
    Simplified version of the network by Recanatesi (2015). A dimensionality
    reduction reduces the number of simulated units.
    """

    def __init__(
            self,
            # Architecture ###########
            num_neurons=100000,
            p=16,
            # Activation #############
            tau=0.01,
            # Gain ###################
            theta=0,
            gamma=0.4,
            # Hebbian rule ###########
            kappa=13000,
            f=0.01,
            # Inhibition #############
            phi_min=0.70,
            phi_max=1.06,
            tau_0=1,
            phase_shift=0.0,
            # Short term association #
            j_forward=1500,
            j_backward=400,
            # Time ###################
            t_tot=450,
            dt=0.001,
            # Noise #####
            xi_0=65,
            # Initialization #########
            r_ini=1,
            first_p=1
    ):

        # Architecture
        self.num_neurons = num_neurons
        self.p = p

        # Activation function
        self.tau = tau

        # Gain function
        self.theta = theta
        self.gamma = gamma

        # Hebbian rule
        self.kappa = kappa
        self.f = f

        # Inhibition
        self.phi_min = phi_min
        self.phi_max = phi_max
        assert phi_max > phi_min
        self.tau_0 = tau_0
        self.phase_shift = phase_shift

        # Short term association
        self.j_forward = j_forward
        self.j_backward = j_backward

        # Time
        self.t_tot = t_tot
        self.dt = dt

        # Noise parameter
        self.xi_0 = xi_0

        # Initialization
        self.r_ini = r_ini
        self.first_p = first_p

        # General pre-computations
        self.t_tot_discrete = int(self.t_tot / self.dt)
        self.phi = np.zeros(self.t_tot_discrete)
        self.relative_excitation = self.kappa / self.num_neurons

        # Unique simplified network attributes
        memory_patterns = \
            np.random.choice([0, 1], p=[1 - self.f, self.f],
                             size=(self.p, self.num_neurons))
        unique_patterns_t, self.num_neurons_per_pattern = \
            np.unique(memory_patterns, axis=1, return_counts=True)
        self.unique_patterns = unique_patterns_t.T
        self.s = self.num_neurons_per_pattern / self.num_neurons

        # print(self.num_neurons_per_pattern)
        # print('n pop', len(self.s))
        # print('s', self.s)

        self.num_populations = len(self.unique_patterns)
        assert self.num_populations == len(self.s)

        self.noise_amplitudes = np.zeros(self.num_populations)
        self.noise_values = np.zeros((self.num_populations,
                                      self.t_tot_discrete))

        self.connectivity_matrix = np.zeros((
            self.num_populations, self.num_populations))

        self.sam_connectivity = np.zeros((self.num_populations,
                                          self.num_populations))

        self.population_currents = np.zeros(self.num_populations)

        # Plotting
        self.average_firing_rate = np.zeros((self.p, self.t_tot_discrete))

    def _present_pattern(self, memory):
        """
        Activation vector will be one pattern for one memory multiplied
        by a parameter whose default value is usually 1.
        """

        self.population_currents[:] = \
            self.unique_patterns[:, memory] * self.r_ini

    def _compute_gaussian_noise(self):
        """Amplitude of uncorrelated Gaussian noise is its variance"""
        print("Computing uncorrelated Gaussian noise...")
        self.amplitude = self.xi_0 * self.s * self.num_neurons

        for i in range(self.num_populations):

            self.noise_values[i] = \
                np.random.normal(loc=0,
                                 scale=(self.xi_0 *
                                        self.num_neurons_per_pattern)**0.5,
                                 size=self.t_tot_discrete)

    def _compute_connectivity_matrix(self):
        """
        Weights do not change in this network architecture and can therefore
        be pre-computed.
        """
        print("Computing connectivity matrix...")

        for v in tqdm(range(self.num_populations)):
            for w in range(self.num_populations):

                sum_ = 0
                for mu in range(self.p):
                    sum_ += \
                        (self.unique_patterns[v, mu] - self.f) \
                        * (self.unique_patterns[w, mu] - self.f)

                self.connectivity_matrix[v, w] = \
                    self.relative_excitation * sum_

                if v == w:
                    self.connectivity_matrix[v, w] = 0

    def _compute_sam_connectivity(self):

        print("Computing SAM connectivity...")

        for v in tqdm(range(self.num_populations)):

            for w in range(self.num_populations):

                sum_forward = 0
                for mu in range(self.p - 1):
                    sum_forward += \
                        self.unique_patterns[v, mu] \
                        * self.unique_patterns[w, mu + 1]

                sum_backward = 0
                for mu in range(1, self.p):
                    sum_backward += \
                        self.unique_patterns[v, mu] \
                        * self.unique_patterns[w, mu - 1]

                self.sam_connectivity[v, w] = \
                    self.j_forward * sum_forward \
                    + self.j_backward * sum_backward

    def _update_activation(self, t):
        """
        Method is iterated every time step and will compute for every pattern.
        Current of population v at time t+dt results from the addition of the
        three terms 'first_term', 'second_term', 'third_term'.
        """

        # new_current = np.zeros(self.activation.shape)

        for v in range(self.num_populations):

            # current = self.activation[population]
            current = self.population_currents[v]

            # First
            first_term = current * (1 - self.dt)  # YES
            # print(first_term)

            # Second
            sum_ = 0
            for w in range(self.num_populations):
                sum_ += \
                    (self.connectivity_matrix[v, w]
                     - self.phi[t] * self.p
                     + self.sam_connectivity[v, w]) \
                    * self.s[w] \
                    * self._g(self.population_currents[w])
                # print(self._g(self.population_currents[w]))
                # print(sum_)

            second_term = sum_ * self.dt  # YES

            # Third
            # print("here", self.noise_values.shape, v, w)
            third_term = self.noise_values[v, t] * self.dt  # YES

            self.population_currents[v] =\
                first_term + second_term + third_term

        # self.activation[:] = new_current
        # self.population_currents[v] = new_current

    def initialize(self, pattern=0):

        self._compute_phi()
        self._compute_connectivity_matrix()
        self._present_pattern(memory=pattern)
        self._compute_gaussian_noise()

    def _save_fr(self, t):

        for mu in range(self.p):

            neurons = self.unique_patterns[mu, :]

            encoding = np.nonzero(neurons)[0]

            v = np.zeros(len(encoding))
            for i, n in enumerate(encoding):
                v[i] = self._g(self.population_currents[n])

            try:
                if len(v):
                    mean = np.mean(v)
                else:
                    mean = 0
            except FloatingPointError:
                mean = 0

            self.average_firing_rate[mu, t] = mean

    @staticmethod
    def _sinusoid(min_, max_, period, t, phase_shift, dt=1.):
        """
        Phi is a sinusoid function related to neuron inhibition.
        It follows the general sine wave function:
            f(x) = amplitude * (2 * pi * frequency * time)
        In order to adapt to the phi_min and phi_max parameters the amplitude
        and an additive shift term have to change.
        Frequency in discrete time has to be adapted from the given tau_0
        period in continuous time.
        """
        amplitude = (max_ - min_) / 2
        frequency = (1 / period) * dt
        shift = min_ + amplitude  # Moves the wave in the y-axis

        return \
            amplitude \
            * np.sin(2 * np.pi * (t + phase_shift / dt) * frequency) \
            + shift

    def _compute_phi(self):

        print("Computing oscillatory inhibition values...")

        for t in range(self.t_tot_discrete):
            self.phi[t] = self._sinusoid(
                min_=self.phi_min,
                max_=self.phi_max,
                period=self.tau_0,
                t=t,
                phase_shift=self.phase_shift * self.tau_0,
                dt=self.dt
            )

    def _g(self, current):

        if current + self.theta > 0:
            gain = (current + self.theta)**self.gamma
        else:
            gain = 0

        return gain

    def simulate(self):

        print(f"Simulating for {self.t_tot_discrete} time steps...\n")
        for t in tqdm(range(self.t_tot_discrete)):
            # print(t, self.population_currents)..
            self._update_activation(t)
            self._save_fr(t)


def plot_phi(network):

    plt.plot(network.phi)
    plt.title("Phi")
    plt.show()


def plot_noise(network):

    for i in range(network.n_population):
        plt.plot(network.noise_values[i])
    plt.title("Noise")
    plt.show()


def plot_array(network):

    data = network.hidden_currents_history

    fig, ax = plt.subplots()
    im = ax.imshow(data)

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")  # , fontsize=font_size)

    plt.title("Hidden layer currents")

    fig.tight_layout()
    plt.show()


def main(force=False):

    bkp_file = f"bkp/s-recanatesi.p"

    os.makedirs(os.path.dirname(bkp_file), exist_ok=True)

    if not os.path.exists(bkp_file) or force:

        np.random.seed(123)

        # factor = 10**(-4)

        simplified_network = SimplifiedNetwork(
                num_neurons=int(10 ** 5),
                f=0.1,
                p=16,
                xi_0=65,
                kappa=13000,
                tau_0=1,
                gamma=2 / 5,
                phi_min=0.7,
                phi_max=1.06,
                phase_shift=0,
                j_forward=1500,
                j_backward=400,
                first_p=1,
                t_tot=2
            )

        simplified_network.initialize()

        print("Pattern:", simplified_network.unique_patterns[:, 0])
        for i in range(simplified_network.num_populations):
            print(simplified_network.unique_patterns[i], simplified_network.num_neurons_per_pattern[i])

        plot_phi(simplified_network)
        plot_noise(simplified_network)
        simplified_network.simulate()

        pickle.dump(simplified_network, open(bkp_file, "wb"))
    else:
        print("Loading from pickle file...")
        simplified_network = pickle.load(open(bkp_file, "rb"))

    # plot.attractor_networks.plot_phi(simplified_network)
    plot.attractor_networks.plot_average_firing_rate(simplified_network)
    plot.attractor_networks.plot_attractors(simplified_network)


if __name__ == "__main__":
    main(force=True)
