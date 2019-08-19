import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Network:
    """
    Pattern consists of multiple binary vectors representing both the item and
    its different characteristics that can be recalled.
    """
    def __init__(self, n_neurons=100000, p=16, f=0.1, first_p=1):
        self.n_neurons = n_neurons
        self.p = p
        self.f = f
        self.first_p = first_p

        self.weights = np.zeros((self.n_neurons, self.n_neurons))

        self.active_fraction = f

        self.initial_currents = \
            np.random.choice([0, 1],
                             p=[1 - self.active_fraction,
                                self.active_fraction],
                             size=self.n_neurons)
        print(self.initial_currents.shape)
        # print("\nInitial currents:\n", self.initial_currents)
        self.patterns = \
            np.random.choice([0, 1], p=[1 - self.f, self.f],
                             size=(self.p, self.n_neurons))
        # print("\nPatterns:\n", self.patterns)

        self.currents = np.copy(self.initial_currents)
        # TODO get rid of self.last_currents
        self.last_currents = np.copy(self.initial_currents)
        self.currents_history = np.copy(self.initial_currents)

    # def present_pattern(self, item):
    #     kanji = item["kanji"]
    #     meaning = item["meaning"]
    #
    #     self.patterns.append(np.concatenate((kanji, meaning), axis=None))

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
        # sum_ = 0
        #
        # for i in range(self.n_neurons):
        #     sum_ += (self.weights[neuron, i] * self.last_currents[i])
        # print(self.currents_history)

        sum_ = np.dot(self.weights[neuron], self.last_currents)

        self.currents[neuron] = self._activation_function(sum_)

        # print(f"\nNeuron {neuron} updated value: {self.currents[neuron]} (sum "
        #       f"was {sum_})")

        # Can also be calculated as:
        # np.dot(self.weights[initial_node, :], self.pattern3)
        # print(sum_)

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
            i += 1
            print(f"Update {i} finished.")

        print(f"\nFinished as attractor {self.currents} after {i} "
              f"node value updates.")

    def simulate(self):
        # assert self.patterns
        # assert self.n_neurons == self.patterns[0].size

        # self._initialize()
        self._compute_weights()
        self._update_all_neurons()
        self._find_attractor()


def plot(network):

    data = network.currents_history

    fig, ax = plt.subplots()
    im = ax.imshow(data)
    ax.set_aspect("auto")

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")  # , fontsize=font_size)

    plt.title("Hidden layer currents")

    fig.tight_layout()
    plt.show()


def main():
    np.random.seed(1234)

    flower = {"kanji": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
              "meaning": np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1])}

    leg = {"kanji": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
           "meaning": np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 1])}

    eye = {"kanji": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
           "meaning": np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])}

    network = Network(
                        n_neurons=6250,
                        f=0.1,
                        # first_p=1
                        p=1
                     )

    # network.present_pattern(flower)
    # network.present_pattern(leg)
    # network.present_pattern(eye)

    network.simulate()
    plot(network)


if __name__ == '__main__':
    main()

# TODO pickle
# TODO plot y axis reformat
# TODO plot initial pattern
# TODO plot attractor
# TODO plot weights matrix
