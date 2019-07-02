import numpy as np

from model_simulation_carlos import n_item

np.random.seed(123)


def main():
    """
    Example one layer neural network with 4 fully connected and recurrent
    units and two output neurons
    """

    # neuron0 = Neuron(neuron_id=0, input_neurons=[0])
    # neuron1 = Neuron(neuron_id=1, input_neurons=[1])
    # neuron2 = Neuron(neuron_id=2, input_neurons=[2])
    # neuron3 = Neuron(neuron_id=3, input_neurons=[3])
    # neuron4 = Neuron(neuron_id=4, input_neurons=[0, 1, 2, 3, 4], role="output")
    # neuron5 = Neuron(neuron_id=5, input_neurons=[0, 1, 2, 3, 5], role="output")
    network = Network()


class Network:
    def __init__(self):
        self.n_neurons = 0

        # Input layer
        try:
            self.n_input_neurons = n_item
        except AttributeError:
            print("No automatic input neurons instantiation used")
        self.create_input_neurons()

    def create_input_neurons(self, verbose=True):
        bits_needed = len(bin(self.n_input_neurons)) - 2
        print(bits_needed)
        for i in range(0, bits_needed):  # Instance n Neurons
            exec("global neuron{0}\nneuron{0} = Neuron()".format(i))
            if verbose:
                print("helpi")

    def create_neuron(self):
        pass

    def predict(self):
        pass

    def train(self):
        pass


class Neuron(Network):
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

    def __init__(self, neuron_id=0, tau=0, theta=0, gamma=0, input_neurons=None,
                 role="hidden", current=0):
        # TODO "=0" in the arguments is only for debugging!
        #super().__init__()

        self.neuron_id = neuron_id
        self.input_currents = None
        self.weights = None
        self.gain = np.empty(0)

        # Update network number of neurons
        # self.network = network
        super().n_neurons += 1

        self.input_neurons = input_neurons

        # Neuron role
        roles = ["input", "hidden", "output"]
        self.role = role
        # if self.role not in roles:
        #     raise AssertionError("Neuron does not have a valid role")
        assert self.role in roles, "Neuron does not have a valid role"
        if self.input_neurons is None:
            self.role = roles[0]
            print("Neuron instanced as role='input'")

        self.current = current
        self.tau = tau  # ref: 0.01. Decay time
        self.input_neurons = input_neurons

        # Gain function
        self.theta = theta  # ref: 0. Gain function threshold
        self.gamma = gamma  # ref: 0.4. Always < 1; gain function is sublinear

        self.initialize_attributes()

    def initialize_attributes(self):
        self.input_currents = np.zeros(self.input_neurons.size)
        self.weights = np.random.rand(1, self.input_neurons.size)

    def compute_gain(self):
        assert self.role == "input", "Input neurons lack the gain function"
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
