import numpy as np


class Network:
    """
    Modified from web.cs.ucla.edu/~rosen/161/notes/hopfield.html
    """
    def __init__(self, n_neuron=5):
        self.n_neuron = n_neuron

        self.pattern1 = np.array([0, 1, 1, 0, 1])
        self.pattern2 = np.array([1, 0, 1, 0, 1])
        self.pattern3 = np.array([1, 1, 1, 1, 1])

        self.weights = np.zeros((self.n_neuron, self.n_neuron))
        self.node_values = self.pattern3
        self.last_node_values = np.array([1, 1, 1, 1, 1])

    def _compute_weights(self):
        """
        Weight values are calculated without any training.

        For several patterns, calculate the matrix for the first pattern,
        then calculate the value for the second matrix and finally add the
        two matrices together.
        """
        print("\nPattern 1:\n", self.pattern1)

        print("\nPattern 1:\n", self.pattern2)

        for i in range(self.n_neuron):
            for j in range(self.n_neuron):
                self.weights[i, j] = (2 * self.pattern1[i] - 1) \
                                     * (2 * self.pattern1[j] - 1)\
                                     + (2 * self.pattern2[i] - 1) \
                                     * (2 * self.pattern2[j] - 1)
                if i == j:
                    self.weights[i, j] = 0

        print("\nWeights after 2 patterns presented:\n", self.weights)

    def _update_node(self, node):
        """
        If you are updating one node of a Hopfield network, then the values of
        all the other nodes are input values, and the weights from those nodes
        to the updated node as the weights.

        In other words, first you do a weighted sum of the inputs from the
        other nodes, then if that value is greater than or equal to 0, you
        output 1. Otherwise, you output 0
        :param node: int node number
        """
        sum_ = 0

        for i in range(self.n_neuron):
            sum_ += (self.weights[node, i] * self.last_node_values[i])

        if sum_ >= 0:
            self.node_values[node] = 1
        else:
            self.node_values[node] = 0

        print(f"\nNode {node} updated value: {self.node_values[node]} (sum "
              f"was {sum_})")

        # Can also be calculated as:
        # np.dot(self.weights[initial_node, :], self.pattern3)
        # print(sum_)

    def _update_all_nodes(self):
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

        values = np.arange(0, self.n_neuron, 1)
        order = np.random.choice(values, self.n_neuron, replace=False)

        for i in order:
            self._update_node(i)

    def _find_attractor(self):
        """
        Cycling through all the nodes each step is the only way to know when to
        stop updating. If a complete network update does not change any of the
        node values, then you are at an attractor so you can stop.
        """
        i = 1
        while np.sum(self.node_values - self.last_node_values) != 0:
            self.last_node_values = self.node_values
            self._update_all_nodes()
            i += 1
            print(f"Update {i} finished.")

        print(f"\nFinished as attractor {self.node_values} after {i} "
              f"node value updates.")

    def simulate(self):
        self._compute_weights()
        self._update_all_nodes()
        self._find_attractor()


def main():
    np.random.seed(123)

    network = Network()
    network.simulate()


if __name__ == '__main__':
    main()
