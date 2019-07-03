import numpy as np

from model_simulation_carlos import n_item

np.random.seed(123)


def main():
    """
    This example creates a fully connected recurrent network
    :return:
    """
    network = Network()

    # Input layer
    network.create_input_neurons()

    # Hidden layer
    network.create_hidden_neurons(n_hidden=46)

    # Show
    network.show_neurons()


class Network:

    roles = 'input', 'hidden', 'output'

    def __init__(self):

        try:  # Try so it can be used even without the "model_simulation" file
            self.n_items = n_item
        except AttributeError:
            print("No automatic input neurons instantiation used")
        self.input_bits_needed = len(bin(self.n_items)) - 2

        self.neurons = {role: [] for role in self.roles}

        # self.neurons_input = []
        # self.neurons_hidden = []
        # self.neurons_output = []

        self.neuron_id = 0

    def create_input_neurons(self, verbose=False):
        """
        Instances n input neurons where n is the amount of neurons to represent
        the total number of question indices in binary form.

        :param verbose: prints loop iteration number
        """
        for i in range(self.input_bits_needed):  # Instance n Neurons
            self.neurons['input'].append(
                Neuron(neuron_id=self.neuron_id, role="input")
            )  # Each neuron will be instanced inside a list, call them as a[n]
            if verbose:
                print(i)

            self.neuron_id += 1

    # def create_output_neurons(self, input_neurons):
    #     for i in range(0, self.n_output):
    #         self.neurons_output.append(
    #             Neuron(input_neurons, role="output")
    #         )

    def create_hidden_neurons(self, n_hidden=10, verbose=False):

        hidden_neuron_inputs = []

        for i in self.neurons['input']:
            hidden_neuron_inputs.append(i)
        if verbose:
            print(hidden_neuron_inputs)
        for j in range(0, n_hidden):
            hidden_neuron_inputs.append(self.neuron_id + 1)
            self.create_neuron(input_neurons=hidden_neuron_inputs,
                               role="hidden")

        # Output layer
        output_neuron_inputs = []
        for k in self.neurons['hidden']:
            output_neuron_inputs.append(k)
        self.create_neuron(input_neurons=hidden_neuron_inputs,
                           role="output")

    def create_neuron(self, input_neurons=None, role="hidden"):
                      # neuron_id=0):

        self.neurons[role].append(
            Neuron(neuron_id=self.neuron_id,
                   role=role, input_neurons=input_neurons)
        )

        # if role == "input":
        #     self.neurons_input.append(
        #         Neuron(neuron_id=self.neuron_id, role=role)
        #     )
        # elif role == "hidden" and input_neurons is not None:
        #     self.neurons_hidden.append(
        #         Neuron(neuron_id=self.neuron_id,
        #                input_neurons=input_neurons, role="hidden")
        #     )
        # elif role == "output" and input_neurons is not None:
        #     self.neurons_output.append(
        #         Neuron(neuron_id=self.neuron_id,
        #                input_neurons=input_neurons, role=role)
        #     )
        # else:
        #     raise ValueError(
        #         "No input neurons given as the argument to "
        #         "instance a non input layer neuron")

        self.neuron_id += 1

    def show_neurons(self):
        for role in self.roles:
            print(f"Number of {role} neurons: ", len(self.neurons[role]))

    def predict(self):
        pass

    def train(self):
        pass


class Neuron:
    """
    TODO cleanup docstring{
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
    }

    :param input_neurons: list of neuron_id connected to this neuron
    :param tau: decay time
    :param theta: gain function threshold
    :param gamma: always < 1; gain function is sublinear

    Note: default argument values are the reference values in the article
    """

    def __init__(self, neuron_id, role, tau=0.01, theta=0, gamma=0.4,
                 input_neurons=None, current=0):

        self.neuron_id = neuron_id

        self.input_neurons = input_neurons
        if self.input_neurons is not None:
            if self.neuron_id in self.input_neurons:
                self.recurrent = True
            else:
                self.recurrent = False

        self.input_currents = None
        self.weights = None
        self.gain = np.empty(0)

        # Neuron role
        roles = ["input", "hidden", "output"]
        self.role = role
        assert self.role in roles, "Neuron does not have a valid role"
        # if self.input_neurons is None and self.role is not "input":
        #     self.role = roles[0]
        #     print("Neuron instanced as role='input'")

        self.current = current
        self.tau = tau
        self.input_neurons = input_neurons

        # Gain function
        self.theta = theta
        self.gamma = gamma

        self.initialize_attributes()

        self.print_attributes()

    def initialize_attributes(self):
        if self.role is not "input":
            self.input_currents = np.zeros(len(self.input_neurons))
            self.weights = np.random.rand(1, len(self.input_neurons))

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

    def print_attributes(self):
        print(
            ', '.join("%s: %s\n" % item for item in vars(self).items())
        )


if __name__ == "__main__":
    main()
