import numpy as np


class Network:
    def __init__(self, n_neurons=5):
        self.n_neurons = n_neurons

        self.pattern1 = np.array([0, 1, 1, 0, 1])
        self.pattern2 = np.array([1, 0, 1, 0, 1])
        self.pattern3 = np.array([1, 1, 1, 1, 1])

        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        self.node_values = self.pattern3
        self.last_node_values = np.array([1, 1, 1, 1, 1])

    def compute_weights(self):
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                self.weights[i, j] = (2 * self.pattern1[i] - 1) \
                                     * (2 * self.pattern1[j] - 1)\
                                     + (2 * self.pattern2[i] - 1) \
                                     * (2 * self.pattern2[j] - 1)
                if i == j:
                    self.weights[i, j] = 0

        print("\nWeights after 2 patterns presented:\n", self.weights)

    def _update_node(self, node):
        sum_ = 0

        for i in range(self.n_neurons):
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

    def update_all_nodes(self):
        """TODO node should be chosen at random"""

        for i in range(self.n_neurons):
            self._update_node(i)

    def find_attractor(self):
        i = 1
        while np.sum(self.node_values - self.last_node_values) != 0:
            self.last_node_values = self.node_values
            self.update_all_nodes()
            i += 1
            print(f"Update {i} finished.")

        print(f"\nFinished as attractor {self.node_values} after {i} "
              f"node value updates.")

    def simulate(self):
        self.compute_weights()
        self.update_all_nodes()
        self.find_attractor()


def main():

    network = Network()
    # network.compute_weights()
    # network.update_all_nodes()
    # network.find_attractor()
    network.simulate()


if __name__ == '__main__':
    main()
