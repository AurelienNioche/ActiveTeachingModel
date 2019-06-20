import numpy as np


def main():
    pass


class Network:
    def __init__(self):
        self.connectivity_matrix = None
        pass

    def connect(self):
        raise NotImplementedError

    def predict(self):
        pass

    def train(self):
        pass


class FullyConnectedNetwork(Network):
    """
    Example one layer neural network with 4 fully connected and recurrent
    units and two output neurons
    """
    def __init__(self):
        self.neuron_layout = np.array([1, 1, 1, 1],
                                      [1, 1, 0, 0])
        # Matrix defined this way to be graphical #
        super().__init__()

    def connect(self, neuron_position):
        if neuron_position * self.neuron_layout == 0:
            print("ERROR: neuron in layer " + neuron_position[0]
                  + " and position " + neuron_position[1]
                  + " in layer does not exist and cannot be connected.")



class Neuron:
    """
    The dynamics of neuron i is represented by the equation:
    τc˙i(t)=−ci(t)+∑j=1NJij ⋅ rj(t)+ξi(t)    (1)
    ri=g(ci)     (2)

    [[t * c(t+1) = -c_{i}(t) + \sum_{j = 1}^{N}{J_{i, j}} * r_{j}{t} + E_{i}(t)]]
    where c, r are respectively the synaptic currents and the firing rates,
    J the connectivity matrix, each ξi is an independent random variable having
    a gaussian distribution with mean zero and variance ξ0 and τ is a constant

    c(t) = c(t-1) * (THE_REST) because if you compute the derivative it gives
    you the original equation from the paper. After that you have the problem
    of equation 2 in the paper and that is solved using by saying that
    the g function...
    """

    def __init__(self, neuron_id, neuron_position, connectivity_matrix, tau, current=0):
        self.neuron_id = neuron_id
        self.neuron_position = neuron_position
        self.connectivity_matrix = connectivity_matrix
        self.current = current
        self.tau = tau

    def activation(self):
        self.current = self.current * (-self.current + connectivity_matrix)


if __name__ == "__main__":
    neuron0 = Neuron(neuron_id=0, neuron_position=[0, 0])
    main()
