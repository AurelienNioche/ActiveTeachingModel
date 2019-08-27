import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import plot.attractor_networks


class Network:
    """
    Pattern consists of multiple binary vectors representing both the item and
    its different characteristics that can be recalled.
    :param learning_rate: float proportion of the theoretical weights learn per
        time step
    """
    def __init__(self, num_neurons=100000, p=16, f=0.1, inverted_fraction=0.3,
                 noise_variance=65, first_p=0, learning_rate=0.3):
        self.num_neurons = num_neurons
        self.p = p
        self.f = f
        self.first_p = first_p
        self.inverted_fraction = inverted_fraction
        self.noise_variance = noise_variance
        self.learning_rate = learning_rate
        assert self.learning_rate <= 1

        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        self.next_theoretical_weights = np.zeros_like(self.weights)
        self.next_weights = np.zeros_like(self.weights)
        self.weights_mean = []

        self.p_recall_history = []

        self.active_fraction = f

        self.initial_currents = np.zeros(self.num_neurons)

        self.patterns = \
            np.random.choice([0, 1], p=[1 - self.f, self.f],
                             size=(self.p, self.num_neurons))
        print("\nPatterns:\n", self.patterns)

        self.currents = np.zeros((1, self.num_neurons), dtype=int)
        self.patterns_evolution = None

        self.question_pattern = np.zeros(self.num_neurons)

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
        print("\nDistorted pattern (i.e. initial currents)...\n", pattern,
              "\n ...in positions\n", idx_reassignment)
        return pattern

    def _initialize_currents(self):
        """Initial currents are set to the distorted pattern."""

        self.currents = np.copy(self.distort_pattern(
            self.patterns[self.first_p],
            self.inverted_fraction)
        )

        # print("\nInitial currents:\n", self.currents)

    def calculate_next_weights(self, pattern):
        """
        Calculate the weights after the presentation of a new pattern but does
        not change the current weights of the network
        """
        for i in tqdm(range(self.num_neurons)):
            for j in range(self.num_neurons):
                if j >= i:
                    break

                self.next_theoretical_weights[i, j] += (2 * pattern[i] - 1) \
                                                       * (2 * pattern[j] - 1) \

        self.next_theoretical_weights += self.next_theoretical_weights.T

    def update_weights(self, weights):
        self.weights += weights

    def compute_weights_all_patterns(self):
        print(f"\n...Computing weights for all patterns...\n")

        for p in tqdm(range(len(self.patterns))):
            self.calculate_next_weights(self.patterns[p])
            self.update_weights(self.next_theoretical_weights)

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
        dot_product = np.dot(self.weights[neuron], self.currents[-2])

        # Amplitude-modulated Gaussian noise
        noise = np.random.normal(loc=0, scale=self.noise_variance**0.5) * 0.05

        self.currents[-1, neuron] = self._activation_function(dot_product
                                                              + noise)

    def _update_all_neurons(self):
        """
        Neurons are updated update in random order as described by Hopfield.
        The full network should be updated before the same node gets updated
        again.
        """
        # self.last_currents = self.currents

        values = np.arange(0, self.num_neurons, 1)
        update_order = np.random.choice(values,
                                        self.num_neurons,
                                        replace=False)

        self.currents = np.vstack((self.currents, np.zeros(self.num_neurons)))

        for i in update_order:
            self._update_current(i)

        # self.currents_history = np.vstack((self.currents_history,
        #                                    self.currents))

    def _compute_patterns_evolution(self):

        for p in range(self.p):
            similarity = np.sum(self.currents[-1] == self.patterns[p])
            self.patterns_evolution = \
                np.vstack((self.patterns_evolution, similarity))

        self.patterns_evolution = self.patterns_evolution.T
        self.patterns_evolution = self.patterns_evolution[0, 1:]

    def _find_attractor(self):
        """
        If an update does not change any of the node values, the networks
        rests at an attractor and updating can stop.
        """
        tot = 1

        while (self.currents[-1] != self.currents[-2]).all() or tot < 2:  # np.sum(self.currents - self.last_currents) != 0:
            self._update_all_neurons()
            self._compute_patterns_evolution()
            tot += 1
            print(f"\nUpdate {tot} finished.\n")

        attractor = np.int_(np.copy(self.currents[-1]))

        print(f"\nFinished as attractor {attractor} after {tot} "
              f"node value updates.\n")

    def simulate(self):
        # assert self.patterns
        # assert self.num_neurons == self.patterns[0].size

        # self._initialize()
        self.compute_weights_all_patterns()
        self._initialize_currents()
        self._update_all_neurons()
        self._find_attractor()

    def learn(self, item=None, time=None):
        """Experimental implementations of a learning rate"""
        # assert (self.weights != self.next_theoretical_weights).all()
        self.next_weights = (self.next_theoretical_weights - self.weights) \
            * self.learning_rate
        print(self.next_weights)

        self.update_weights(self.next_weights)

        self.weights_mean.append(-np.mean(self.next_theoretical_weights)\
                                 + np.mean(self.weights))

        # plot.attractor_networks.plot_weights(self)

    def fully_learn(self):
        tot = 1

        while (self.weights[-1] != self.next_theoretical_weights).all():
            self.learn()
            tot += 1

        print(f"\nFinished learning after {tot} "
              f"node weight updates.\n")

    def simulate_learning(self):
        self.calculate_next_weights(self.patterns[self.first_p])
        # self._initialize_currents()
        self.fully_learn()
        # self._update_all_neurons()
        # self._find_attractor()

    def p_recall(self, item, time=None):
        """
        after choosing, compare the chosen pattern with the correct pattern
        to retrieve the probability of recall
        """

        # bin_question is the partial (distorted) array
        question_array = np.array([item])
        bin_question = ((question_array[:, None]
                         & (1 << np.arange(8))) > 0).astype(int)
        bin_question = np.append(bin_question, np.zeros(self.num_neurons
                                                        - bin_question.size))

        print("Item given as pattern:", bin_question)

        self.currents = np.vstack((self.currents, bin_question))
        self._update_all_neurons()

        match = np.sum(self.currents[-1] == bin_question)
        p_r = match / self.num_neurons
        print("\nCurrent after item presentation and one update:\n",
              self.currents[-1])
        print("\nProbability of recall of the item: ", p_r)

        return p_r


def plot_average_firing_rate(network):
    """TODO make it work and move it to /plot"""

    data = network.patterns_evolution
    n_iteration = network.currents.shape[0] - 1
    # print(n_iteration)

    x = np.arange(0, n_iteration, dtype=float)
    print("x", x.size)

    fig, ax = plt.subplots()

    ax.plot(x, data, linewidth=0.5, alpha=1)

    try:
        for i in range(data.shape[0]-1):
            ax.plot(x, data[i+1], linewidth=0.5, alpha=1)
            if i > 1:
                break
    except:
        print("1")

    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel("Average firing rate")
    plt.show()


def main(force=False):

    bkp_file = f"bkp/hopfield.p"

    os.makedirs(os.path.dirname(bkp_file), exist_ok=True)

    if not os.path.exists(bkp_file) or force:

        np.random.seed(123)

        # flower = {"kanji": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        #           "meaning": np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1])}
        #
        # leg = {"kanji": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        #        "meaning": np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 1])}
        #
        # eye = {"kanji": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        #        "meaning": np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])}

        network = Network(
                            num_neurons=200,
                            f=0.7,
                            p=1,
                            inverted_fraction=0.2,
                            learning_rate=0.00075
                         )

        # network.present_pattern(flower)
        # network.present_pattern(leg)
        # network.present_pattern(eye)

        # network.simulate()

        # network.p_recall(12)

        # print(network.weights)
        # network._initialize_currents()
        # network._update_all_neurons()
        # for i in range(0):
        #     network._find_attractor()
        print(network.weights)
        print(network.patterns)
        network.calculate_next_weights(network.patterns[0])
        network._update_all_neurons()
        print(network.next_theoretical_weights)

        for i in range(10):
            print(np.mean(network.weights))
            network.learn()
            network._update_all_neurons()
            match = np.sum(network.currents[-1] == network.patterns[0])
            p_recall = match / network.num_neurons
            network.p_recall_history.append(p_recall)

        print(network.weights)

        print(np.mean(network.weights))

        plot.attractor_networks.plot_mean_weights(network)
        plot.attractor_networks.plot_p_recall(network)

        pickle.dump(network, open(bkp_file, "wb"))
    else:
        print("Loading from pickle file...")
        network = pickle.load(open(bkp_file, "rb"))

    plot.attractor_networks.plot_currents(network)
    # plot.attractor_networks.plot_weights(network)
    # plot_average_firing_rate(network)


if __name__ == '__main__':
    main(force=True)
