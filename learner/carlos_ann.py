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
    neuron4 = Neuron(neuron_id=4, input_neurons=[0, 1, 2, 3, 4], role="output")
    neuron5 = Neuron(neuron_id=5, input_neurons=[0, 1, 2, 3, 5], role="output")
    network = Network


class Network:
    def __init__(self):
        self.connectivity_matrix = None
        self.n_neurons = 0

    @staticmethod
    def instance_input_neurons(n=15, verbose=False):
        for i in range(n):
            exec("global neuron{} = Neuron()".format(i))
            if verbose:
                print(i)

    def predict(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class Neuron:
    """
    The dynamics of neuron i is represented by the equation:
    τc˙i(t)=−ci(t)+∑j=1NJij ⋅ rj(t)+ξi(t)   (1)
    ri=g(ci)     (2)

    [[t * c(t+1) = -c_{i}(t) + \sum_{j = 1}^{N}{J_{i, j}} * r_{j}{t}
    + E_{i}(t)]]
    where c, r are respectively the synaptic currents and the firing rates,
    J the connectivity matrix, each ξi is an independent random variable having
    a gaussian distribution with mean zero and variance ξ0 and τ is a constant

    c(t) = c(t-1) * (THE_REST) because if you compute the derivative it gives
    you the original equation from the paper. After that you have the problem
    of equation 2 in the paper and that is solved using by saying that
    the g function...
    """

    def __init__(self, neuron_id, tau, theta, gamma, network="network", input_neurons=None,
                 role="hidden", current=0):
        self.neuron_id = neuron_id
        self.input_currents = None
        self.weights = None
        self.gain = np.empty(0)

        self.network = network
        network.n_neurons += 1

        self.input_neurons = input_neurons

        # Neuron role
        roles = ["input", "hidden", "output"]
        self.role = role
        if self.role not in roles:
            raise AssertionError("Neuron does not have a valid role")
        if self.input_neurons is None:
            self.role = roles[0]
            print("Neuron instanced as role='input'")

        self.current = current
        self.tau = tau
        self.input_neurons = input_neurons

        # Gain function
        self.theta = theta
        self.gamma = gamma  # always < 1; gain function is sublinear

        self.initialize_attributes()

    def initialize_attributes(self):
        self.input_currents = np.zeros(self.input_neurons.size)
        self.weights = np.random.rand(1, self.input_neurons.size)

    def compute_gain(self):
        if self.role == "input":
            raise AssertionError("Input neurons lack the gain function")
        for i in self.input_currents:
            if i + self.theta > 0:
                new_gain = (i + self.theta) ** self.gamma
                self.gain = np.concatenate(self.gain, new_gain)
            else:
                self.gain = np.concatenate(self.gain, 0)

    def activation(self):
        self.current =\
            self.current * (-self.current + self.input_currents * self.weights
                            * self.gain + np.random.rand() / self.tau)


if __name__ == "__main__":
    main()
