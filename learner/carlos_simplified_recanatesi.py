import os
import pickle

import numpy as np
from tqdm import tqdm

import plot.attractor_networks
from learner.carlos_recanatesi import Network

np.seterr(all='raise')


class SimplifiedNetwork(Network):
    """
    Simplified version of the network by Recanatesi (2015). A dimensionality
    reduction reduces the number of simulated units.
    """
    def __init__(self):
        super().__init__()

        self.connectivity = np.unique(self.connectivity, axis=1)

        self.neuron_populations = self.connectivity.shape[1]

        self.weights_constant = np.zeros((
            self.neuron_populations, self.neuron_populations))
        self.activation = np.zeros(self.neuron_populations)

    def _present_pattern(self):

        self.activation[:] = self.connectivity[self.first_p, :] \
                             * self.r_ini

    def _gaussian_noise(self):
        """
        Formula simplified from paper as N gets cancelled from when xi_v and
        S_v equations are put together.
        """
        return np.random.normal(loc=0, scale=(self.xi_0
                                              * np.unique(self.connectivity,
                                                          axis=1).shape[1])
                                ** 0.5)

    def _compute_weight_constants(self):

        print("Computing simplified weight constants...")
        for i in tqdm(range(self.neuron_populations)):
            for j in range(self.neuron_populations):

                sum_ = 0
                for mu in range(self.p):
                    sum_ += \
                        (self.connectivity[mu, i] - self.f) \
                        * (self.connectivity[mu, j] - self.f)

                self.weights_constant[i, j] = \
                    self.kappa_over_n * sum_

    def _update_activation(self):

        new_current = np.zeros(self.activation.shape)

        for i in range(self.neuron_populations):

            current = self.simplified_activation[i]
            noise = self.simplified_gaussian_noise()  # correct
            fraction_neurons = self.neuron_populations / self.n_neurons
            # correct

            sum_ = 0
            for j in range(self.neuron_populations):
                sum_ += \
                    (self.weights_constant[i, j]
                     - self.kappa_over_n * self.phi
                     + self.delta_weights[i, j]) \
                    * self._g(self.activation[j]) \
                    * fraction_neurons

            business = sum_ + noise

            second_part = business * self.dt  # done

            first_part = current * \
                (1 - self.dt)  # done

            new_current[i] = first_part + second_part

        self.activation[:] = new_current

    def _initialize(self):

        self._compute_weight_constants()
        self._present_pattern()


def main(force=False):

    bkp_file = f'bkp/hopfield.p'

    os.makedirs(os.path.dirname(bkp_file), exist_ok=True)

    if not os.path.exists(bkp_file) or force:

        np.random.seed(1234)

        factor = 10**(-3)

        network = Network(
                n_neurons=int(10**5*factor),
                f=0.1,
                p=16,
                xi_0=65,
                kappa=13000,
                tau_0=1,
                gamma=2 / 5,
                phi_min=0.7,
                phi_max=1.06,
                phase_shift=0.75,
                j_forward=1500*factor,
                j_backward=400*factor,
                first_p=0,
                t_tot=4,
            )

        network.simulate()

        pickle.dump(network, open(bkp_file, 'wb'))
    else:
        print('Loading from pickle file...')
        network = pickle.load(open(bkp_file, 'rb'))

    plot.attractor_networks.plot_phi(network)
    plot.attractor_networks.plot_average_firing_rate(network)
    plot.attractor_networks.plot_attractors(network)


if __name__ == "__main__":
    main(force=True)
