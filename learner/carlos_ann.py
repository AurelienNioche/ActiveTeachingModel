import numpy as np

np.random.seed(123)


def main():
    """
    Example one layer neural network with 4 fully connected and recurrent
    units and two output neurons
    """
    neuron0 = Neuron(neuron_id=0, input_neurons=[0])
    neuron1 = Neuron(neuron_id=1, input_neurons=[1])
    neuron2 = Neuron(neuron_id=2, input_neurons=[2])
    neuron3 = Neuron(neuron_id=3, input_neurons=[3])
    neuron4 = Neuron(neuron_id=4, input_neurons=[0, 1, 2, 3, 4])
    neuron5 = Neuron(neuron_id=5, input_neurons=[0, 1, 2, 3, 5])


class Network:
    def __init__(self):
        self.connectivity_matrix = None
        self.n_neurons = 0

    # def connect(self, neuron_position):
    #     if neuron_position * self.neuron_layout == 0:
    #         raise LookupError("ERROR: neuron in layer " + neuron_position[0]
    #               + " and position " + neuron_position[1]
    #               + " in layer does not exist and cannot be connected.")

    def predict(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


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

    def __init__(self, neuron_id, tau, network="network", input_neurons=None,
                 current=0):
        self.neuron_id = neuron_id
        self.inputs = None
        self.weights = None

        self.network = network
        network.n_neurons += 1

        self.input_neurons = input_neurons

        # roles = ["input", "hidden", "output"]
        # self.role = None
        # if self.input_neurons is None:
        #     self.role = roles[0]
        # elif self.output_neurons is None:
        #     self.role = roles[2]
        # elif self.input_neurons is not None and self.output_neurons is not None:
        #     self.role = roles[1]
        # else:
        #     raise AssertionError("Neuron does not have a valid role")

        self.current = current
        self.tau = tau
        self.input_neurons = input_neurons

    def initialize_attributes(self):
        self.inputs = np.zeros(self.input_neurons.size)
        self.weights = np.random.rand(1, self.input_neurons.size)

    def activation(self):
        self.current = self.current * (-self.current + self.inputs * self.weights)


if __name__ == "__main__":
    main()

# TODO after long weekend please continue working in the activation function
