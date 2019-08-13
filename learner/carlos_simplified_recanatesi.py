import pickle
import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from learner.carlos_recanatesi import Network

np.seterr(all='raise')


class SimplifiedNetwork(Network):
    """
    Simplified version of the network by Recanatesi (2015). A dimensionality
    reduction reduces the number of simulated units.
    """
    def __init__(self):
        super.__init__()

        self.simplified_connectivity = np.unique(self.connectivity, axis=1)

        self.neuron_populations = self.simplified_connectivity.shape[1]

        self.simplified_weights_constant = np.zeros((
            self.neuron_populations, self.neuron_populations))
        self.simplified_activation = np.zeros(self.neuron_populations)

    def _present_pattern(self):

        self.simplified_activation[:] = self.connectivity[self.first_p, :] \
                             * self.r_ini

    def gaussian_noise(self):
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
                        (self.simplified_connectivity[mu, i] - self.f) \
                        * (self.simplified_connectivity[mu, j] - self.f)

                self.simplified_weights_constant[i, j] = \
                    self.kappa_over_n * sum_

    def _initialize(self):

        self._compute_simplified_weight_constants()