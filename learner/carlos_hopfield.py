import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm


class Network:
    """
    Pattern consists of multiple binary vectors representing both the item and
    its different characteristics that can be recalled.
    """
    def __init__(self, n_neurons=100000, p=16, f=0.1, inverted_fraction=0.3,
                 first_p=1):
        self.n_neurons = n_neurons
        self.p = p
        self.f = f
        self.first_p = first_p
        self.inverted_fraction = inverted_fraction

        self.weights = np.zeros((self.n_neurons, self.n_neurons))

        self.active_fraction = f

        self.initial_currents = np.zeros(self.n_neurons)
            # np.random.choice([0, 1],
            #                  p=[1 - self.active_fraction,
            #                     self.active_fraction],
            #                  size=self.n_neurons)

        self.patterns = \
            np.random.choice([0, 1], p=[1 - self.f, self.f],
                             size=(self.p, self.n_neurons))
        print("\nPatterns:\n", self.patterns)

        self.currents = np.zeros(self.n_neurons)
        # TODO get rid of self.last_currents
        self.last_currents = np.zeros(self.n_neurons)
        self.currents_history = np.zeros(self.n_neurons)
        self.patterns_evolution = None

    # def present_pattern(self, item):
    #     kanji = item["kanji"]
    #     meaning = item["meaning"]
    #
    #     self.patterns.append(np.concatenate((kanji, meaning), axis=None))

    @staticmethod
    def distort_pattern(pattern, proportion):
        """
        Inverts the array in random positions proportional to array size.

        :param pattern: array-like binary vector to distort
        :param proportion: float 0 to 1, 1 being full array inversion

        :return pattern: array-like binary vector with inverted elements
        """

        num_inversions = int(pattern.size * proportion)
        assert proportion != 1
        idx_reassignment = np.random.choice(pattern.size, num_inversions,
                                            replace=False)
        pattern[idx_reassignment] = np.invert(pattern[idx_reassignment] - 2)
        print("\nDistorted pattern...\n", pattern,
              "\n ...in positions\n", idx_reassignment)
        return pattern

    def _initialize_currents(self):

        self.initial_currents = self.distort_pattern(self.patterns[0],
                                                     self.inverted_fraction)

        print("\nInitial currents:\n", self.initial_currents)

        self.currents = np.copy(self.initial_currents)
        self.last_currents = np.copy(self.initial_currents)
        self.currents_history = np.copy(self.initial_currents)

    def _compute_weights(self):
        """
        Weight values are calculated without any training.

        For several patterns, calculate the matrix for the first pattern,
        then calculate the value for the second matrix and finally add the
        two matrices together.
        """

        print(f"...Computing weights...")

        # for p in range(len(self.patterns)):
        #     for i in tqdm(range(self.n_neurons)):
        #         for j in range(self.n_neurons):
        #             self.weights[i, j] += (2 * self.patterns[p, i] - 1) \
        #                                  * (2 * self.patterns[p, j] - 1) \
        #
        #             if i == j:
        #                 self.weights[i, j] = 0
        #     print(f"Finished computing for pattern {p}")

        # for p in range(len(self.patterns)):
        #     for i in range(self.n_neurons):
        #         for j in range(self.n_neurons):
        #             self.weights[i, j] += (2*self.patterns[p][i] - self.active_fraction) \
        #                                  * (2*self.patterns[p][j] - self.active_fraction) \
        #
        #             if i == j:
        #                 self.weights[i, j] = 0

        # print("\nWeights after patterns presented:\n", self.weights)

        for p in range(len(self.patterns)):
            for i in tqdm(range(self.n_neurons)):
                for j in range(self.n_neurons):
                    if j >= i:
                        break

                    self.weights[i, j] += (2 * self.patterns[p, i] - 1) \
                        * (2 * self.patterns[p, j] - 1) \

            self.weights += self.weights.T
            print(f"...Finished computing for pattern {p}")

        print("Done!")

    @staticmethod
    def _activation_function(x):
        """Heaviside"""
        return int(x >= 0)

    def _update_current(self, neuron):
        """
        If you are updating one node of a Hopfield network, then the values of
        all the other nodes are input values, and the weights from those nodes
        to the updated node as the weights.

        In other words, first you do a weighted sum of the inputs from the
        other nodes, then if that value is greater than or equal to 0, you
        output 1. Otherwise, you output 0

        :param neuron: int neuron number
        """
        dot_product = np.dot(self.weights[neuron], self.last_currents)

        self.currents[neuron] = self._activation_function(dot_product)

    def _update_all_neurons(self):
        """
        There are two approaches:

        The first is synchronous updating, which means all the nodes get
        updated at the same time, based on the existing state (i.e. not on
        the values the nodes are changing to). To update the nodes in this
        method, you can just multiply the weight matrix by the vector of the
        current state.

        This is not very realistic in a neural sense, as neurons do not all
        update at the same rate. They have varying propagation delays, varying
        firing times, etc. A more realistic assumption would be to update them
        in random order, which was the method described by Hopfield. Random
        updating goes on until the system is in a stable state. Note that the
        full network should be updated before the same node gets updated again.
        """

        values = np.arange(0, self.n_neurons, 1)
        update_order = np.random.choice(values, self.n_neurons, replace=False)

        for i in update_order:
            self._update_current(i)

        self.currents_history = np.vstack((self.currents_history,
                                           self.currents))

    def _compute_patterns_evolution(self):

        for p in range(self.p):
            similarity = np.sum(self.currents == self.patterns[p])
            if not self.patterns_evolution:
                self.patterns_evolution = similarity
            else:
                np.vstack((self.patterns_evolution, similarity))

    def _find_attractor(self):
        """
        Cycling through all the nodes each step is the only way to know when to
        stop updating. If a complete network update does not change any of the
        node values, then you are at an attractor so you can stop.
        """
        # TODO review, always 2 iterations
        i = 1

        assert np.sum(self.currents - self.last_currents) != 0
        while (self.currents != self.last_currents).all():  # np.sum(self.currents - self.last_currents) != 0:
            self.last_currents = self.currents
            self._update_all_neurons()
            self._compute_patterns_evolution()
            i += 1
            print(f"Update {i} finished.")

        print(f"\nFinished as attractor {self.currents} after {i} "
              f"node value updates.")

    def simulate(self):
        # assert self.patterns
        # assert self.n_neurons == self.patterns[0].size

        # self._initialize()
        self._compute_weights()
        self._initialize_currents()
        self._update_all_neurons()
        self._find_attractor()


def plot(network):

    data = network.currents_history

    fig, ax = plt.subplots()
    im = ax.imshow(data)
    ax.set_aspect("auto")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")  # , fontsize=font_size)

    ax.set_title("Network currents history")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Iteration")

    plt.tight_layout()
    plt.show()


def plot_average_firing_rate(network):

    average_fr = network.patterns_evolution
    n_iteration = average_fr.shape[1]
    dt = network.dt

    x = np.arange(n_iteration, dtype=float) * dt

    fig, ax = plt.subplots()

    for i, y in enumerate(average_fr):
        ax.plot(x, y, linewidth=0.5, alpha=1)
        if i > 1:
            break

    ax.set_xlabel('Time (cycles)')
    ax.set_ylabel('Average firing rate')
    plt.show()


def main(force=False):

    bkp_file = f"bkp/hopfield.p"

    os.makedirs(os.path.dirname(bkp_file), exist_ok=True)

    if not os.path.exists(bkp_file) or force:

        np.random.seed(1234)

        # flower = {"kanji": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        #           "meaning": np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1])}
        #
        # leg = {"kanji": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        #        "meaning": np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 1])}
        #
        # eye = {"kanji": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        #        "meaning": np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])}

        network = Network(
                            n_neurons=14,
                            f=0.4,
                            p=2,
                            inverted_fraction=0.9
                         )

        # network.present_pattern(flower)
        # network.present_pattern(leg)
        # network.present_pattern(eye)

        network.simulate()
        pickle.dump(network, open(bkp_file, "wb"))
    else:
        print("Loading from pickle file...")
        network = pickle.load(open(bkp_file, "rb"))

    plot(network)
    # plot_average_firing_rate(network)


if __name__ == '__main__':
    main(force=True)
