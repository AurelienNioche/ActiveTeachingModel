import numpy as np
from tqdm import tqdm

from behavior.data_structure import Task
from learner.generic import Learner


class Network(Learner):

    roles = 'input', 'hidden', 'output'

    def __init__(self, tk, n_epoch=1):

        super().__init__()

        try:  # Try so it can be used even without the "model_simulation" file
            self.n_items = tk.n_item
        except AttributeError:
            print("No automatic input neurons instantiation used")
        self.input_bits_needed = len(bin(self.n_items)) - 2

        self.neurons = {role: [] for role in self.roles}

        self.neuron_id = 0

        # self.train(n_epoch)

    def create_input_neurons(self, verbose=False):
        """
        Instances n input neurons where n is the amount of neurons to represent
        the total number of question indices in binary form.

        :param verbose: prints loop iteration number
        """
        for i in range(self.input_bits_needed):  # Instance n Neurons
            # self.neurons['input'].append(
            #     Neuron(neuron_id=self.neuron_id, role="input")
            # )
            self._create_neuron(role='input')
            # Each neuron will be instanced inside a list, call them as a[n]
            if verbose:
                print(i)

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
            self._create_neuron(input_neurons=hidden_neuron_inputs,
                                role="hidden")

        # Output layer
        output_neuron_inputs = []
        for k in self.neurons['hidden']:
            output_neuron_inputs.append(k)
        self._create_neuron(input_neurons=hidden_neuron_inputs,
                            role="output")

    def _create_neuron(self, input_neurons=None, role="hidden"):
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

    def train(self, n_epochs):
        """
        :param n_epochs: int in range(0, +inf). Number of epochs
        :return: None
        """

        if n_epochs < 0:
            raise ValueError("n_epochs not int or not in range(0, +inf)")

        for _ in tqdm(range(0, n_epochs)):
            for neuron in self.neurons["hidden"]:
                    neuron.compute_gain()
                    neuron.update_current()
            for neuron in self.neurons["output"]:
                    neuron.compute_gain()
                    neuron.update_current()

    def predict(self):
        pass

    def update(self, epochs):
        pass

    def decide(self, question, possible_replies, time=None):
        pass

    def learn(self, question, time=None):
        pass

    def unlearn(self):
        pass

    def _p_choice(self, question, reply, possible_replies, time=None):
        pass

    def _p_correct(self, question, reply, possible_replies, time=None):
        pass

    def p_recall(self, item, time=None):
        pass


class Neuron:
    """
    Modified from Recanatesi (2015) equations 1 to 4.

    :param input_neurons: list of neuron_id connected to this neuron
    :param tau: decay time
    :param theta: gain function threshold
    :param gamma: gain func exponent. Always sub-linear, then value < 1
    :param kappa: excitation parameter
    :param phi: inhibition parameter in range(0.70, 1.06)

    Note: default argument values are the reference values in the article but
    for phi; only phi_min and phi_max are given in Recanatesi (2015).
    """

    def __init__(self, neuron_id, role, tau=0.01, theta=0, gamma=0.4,
                 kappa=13000, phi=1,
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
        self.weights_hebbian = None
        self.gain = None

        # Neuron role
        roles = ["input", "hidden", "output"]
        self.role = role
        assert self.role in roles, "Neuron does not have a valid role"
        # if self.input_neurons is None and self.role is not "input":
        #     self.role = roles[0]
        #     print("Neuron instanced as role='input'")

        # Neuron dynamics
        self.current = current
        self.tau = tau
        self.input_neurons = input_neurons

        # Gain function
        self.theta = theta
        self.gamma = gamma

        # Hebbian rule
        self.kappa = kappa
        self.phi = phi

        self._initialize_attributes()

        for i in range(0, 1):
            if self.role is not "input":
                self.compute_gain()
                self.update_current()
        self.print_attributes()

    @staticmethod
    def _random_small_value():
        number = abs(np.random.normal(loc=0.5, scale=0.25))
        number = max(min(1, number), 0)
        return number

    def _initialize_attributes(self):
        if self.role is "input":
            self.current = np.random.randint(0, 2)
        else:
            self.input_currents = \
                np.array([self._random_small_value()
                          for _ in range(len(self.input_neurons))])
            # self.input_currents = np.random.normal(loc=0.5, scale=0.25,
            # size=len(self.input_neurons))
            # self.input_currents[self.input_currents>1] = 1
            # self.input_currents[self.input_currents<0] = 0
            self.weights = np.random.rand(1, len(self.input_neurons))
            self.current = self._random_small_value()

    def compute_gain(self):
        # assert self.role == "input", "Input neurons lack the gain function"
        # for i in self.input_currents:
        #     if i + self.theta > 0:
        #         new_gain = (i + self.theta) ** self.gamma
        #         self.gain = np.concatenate(self.gain, new_gain)
        #     else:
        #         self.gain = np.concatenate(self.gain, 0)
        if self.current + self.theta > 0:
            self.gain = (self.current + self.theta) ** self.gamma
        else:
            self.gain = 0

    def update_current(self):
        assert self.role is not "input"
        self.weights_hebbian = np.dot(self.current, self.input_currents)
        self.current =\
            self.current * ((-self.current + sum(self.weights_hebbian)
                            * self.gain
                            + self._random_small_value()))
        if self.current < 0.55555:
            self.current = 0
        else:
            self.current = 1
        # / self.tau)

    def print_attributes(self):
        """
        :returns: multi-line string of instance attributes
        """
        print(
            '* '.join("%s: %s\n" % item for item in vars(self).items())
        )


def main():
    """
    This example creates a fully connected recurrent network
    """

    np.random.seed(1234)

    network = Network(tk=Task(t_max=100, n_item=30))

    # Input neurons
    network.create_input_neurons()

    # Hidden neurons
    network.create_hidden_neurons(n_hidden=10)

    # Show
    network.show_neurons()

    spam = network.neurons["output"][0].weights_hebbian
    network.train(200000)
    eggs = network.neurons["output"][0].weights_hebbian
    print("spam ", spam, "\neggs", eggs)
    for i in network.neurons["input"]:
        print(i.current)


if __name__ == "__main__":
    main()
