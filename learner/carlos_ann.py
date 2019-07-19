import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from behavior.data_structure import Task
from learner.generic import Learner

np.seterr(all='raise')


class NetworkParam:
    def __init__(self, n_hidden=40, n_epoch=3):
        self.n_epoch = n_epoch
        self.n_hidden = n_hidden


class Network(Learner):

    version = 0.1
    bounds = ('d', 0.001, 1.0), \
             ('tau', -1, 1), \
             ('s', 0.001, 1)  # TODO change

    roles = 'input', 'hidden', 'output'

    def __init__(self, tk, param, verbose=False):

        super().__init__()

        self.tk = tk
        self.verbose = verbose

        if param is None:
            pass
        elif type(param) == dict:
            self.pr = NetworkParam(**param)
        elif type(param) in (tuple, list, np.ndarray):
            self.pr = NetworkParam(*param)
        else:
            raise Exception(
                f"Type {type(param)} is not handled for parameters")

        try:  # Try so it can be used even without the "model_simulation" file
            self.n_items = tk.n_item
        except AttributeError:
            print("No automatic input neurons instantiation used")
        self.input_bits_needed = len(bin(self.n_items)) - 2

        self.neurons = {role: [] for role in self.roles}
        self.neuron_id = 0
        self.hidden_currents_history = np.zeros((self.pr.n_epoch,
                                                 self.pr.n_hidden))

        self.create_input_neurons()
        self.create_hidden_neurons(n_hidden=self.pr.n_hidden)

        self._train(self.pr.n_epoch)

    def create_input_neurons(self, verbose=False):
        """
        :param verbose: prints loop iteration number.

        Instances n input neurons where n is the amount of neurons to represent
        the total number of question indices in binary form.
        """
        for i in range(self.input_bits_needed):  # Instance n Neurons
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

        self.neuron_id += 1

    def show_neurons(self):
        for role in self.roles:
            print(f"Number of {role} neurons: ", len(self.neurons[role]))

    def _train(self, n_epochs):
        """
        :param n_epochs: int in range(0, +inf). Number of epochs
        """

        if n_epochs < 0:
            raise ValueError("n_epochs not int or not in range(0, +inf)")

        for i in tqdm(range(n_epochs)):
            for neuron in self.neurons["hidden"]:
                neuron.compute_gain()
                neuron.update_current()
                for j, val in enumerate(self.neurons["hidden"]):
                    self.hidden_currents_history[i, j] =\
                        self.neurons["hidden"][j].current

            for neuron in self.neurons["output"]:
                neuron.compute_gain()
                neuron.update_current()

        print(self.hidden_currents_history)

    def p_recall(self, item, time=None):
        p_recall = self.neurons["output"][0].current
        return p_recall

    def _p_choice(self, question, reply, possible_replies=None,
                  time=None, time_index=None):
        """Modified from ActR"""

        success = question == reply

        p_recall = self.neurons["output"][0].current

        # If number of possible replies is defined
        if self.tk.n_possible_replies is not None:
            p_correct = self.p_random + p_recall*(1 - self.p_random)

            if success:
                p_choice = p_correct

            else:
                p_choice = (1-p_correct) / (self.tk.n_possible_replies - 1)

        else:
            # Ignore in computation of reply the alternatives
            # p_choice = p_recall if success else 1-p_recall
            p_correct = self.p_random + p_recall * (1 - self.p_random)

            if success:
                p_choice = p_correct

            else:
                p_choice = (1 - p_correct)

        return p_choice

    def _p_correct(self, question, reply, possible_replies=None,
                   time=None, time_index=None):

        p_correct = self._p_choice(question=question, reply=question,
                                   time=time, time_index=time_index)

        correct = question == reply
        if correct:
            return p_correct

        else:
            return 1-p_correct

    def decide(self, question, possible_replies, time=None,
               time_index=None):

        p_r = self.neurons["output"][0].current
        r = np.random.random()

        if p_r > r:
            reply = question
        else:
            reply = np.random.choice(possible_replies)

        if self.verbose:
            print(f't={self.t}: question {question}, reply {reply}')
        return reply

    def learn(self, question, time=None):
        pass

    def unlearn(self):
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

            # I simplified ============================
            # if self.neuron_id in self.input_neurons:
            #     self.recurrent = True
            # else:
            #     self.recurrent = False
            self.recurrent = self.neuron_id in self.input_neurons

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
        if self.current + self.theta > 0:
            self.gain = (self.current + self.theta) ** self.gamma
        else:
            self.gain = 0

    def update_current(self):
        assert self.role is not "input"
        self.weights_hebbian = np.dot(self.current, self.input_currents)

        r = self._random_small_value()

        try:
            sum_gain = np.sum(self.weights_hebbian) * self.gain

            # I rewrote
            # self.current *= \
            #     ((-self.current + sum(self.weights_hebbian)
            #       * self.gain
            #       + self._random_small_value()))
            self.current *= \
                (-self.current + sum_gain + r)

        except FloatingPointError as e:
            print('gain:', self.gain)
            print()
            print('weights_hebbian', self.weights_hebbian)
            # If you want to continue the execution, comment this line...
            raise e

            # ... and uncomment this one
            # self.current = np.inf

        # if self.role == "hidden":
        self.current = int(self.current > 0.5)
            # TODO find a way to implement output = P(r) w/o breaking script
            # / self.tau)
        if self.role == "output":
            print(self.weights_hebbian)  # Debugging

    def print_attributes(self):
        """
        :returns: multi-line string of instance attributes
        """
        print(
            '* '.join("%s: %s\n" % item for item in vars(self).items())
        )


def plot(network):
    # data = network.hidden_currents_history
    # # data = np.zeros(network.pr.n_hidden)
    # # for i, val in enumerate(network.neurons["hidden"]):
    # #     data[i] = network.neurons["hidden"][i].current
    # # print(data.shape)
    # # factors = []
    # # dimension_pairs = []
    # # for i in range(network.pr.n_hidden):
    # #     if (network.pr.n_hidden % (network.pr.n_hidden - 1)) == 0:
    # #         factors.append()
    # #         # now I have the factors, make it pairs next
    # # numbo[-1]
    # # data = np.reshape(data, (-1, 5))
    # # print(data)
    # # x, y = np.meshgrid(range(data.shape[0]), range(data.shape[1]))
    #
    # # x = np.arange(0, network.pr.n_hidden, 1)
    # # y = np.zeros(1) + 1
    # # z = data
    # # h = plt.contourf(x, y, z)
    #
    # data = np.expand_dims(data, axis=0)
    # # plt.pcolormesh(data)
    # data2 = data * 0
    # # plt.pcolormesh(data2)
    # fig, axs = plt.subplots(nrows=2, ncols=1, subplot_kw={'xticks': [],
    #                                                       'yticks': []})
    #
    # for ax in axs:
    #     ax.imshow(data, cmap='viridis')
    #     ax.imshow(data2, cmap='viridis')
    #
    # plt.tight_layout()
    # plt.show()
    # # z = data
    # # c = plt.ax.contourf(x, y, z)  # cmap='viridis')
    # # divider = make_axes_locatable(ax)
    # # cax = divider.append_axes("right", size="5%", pad=0.05)
    # # plt.colorbar(c, cax=cax, ticks=y_ticks)

    print(network.hidden_currents_history)
    plt.imshow(network.hidden_currents_history)


def main():
    """
    This example creates a fully connected recurrent network
    """

    np.random.seed(1234)

    network = Network(tk=Task(t_max=100, n_item=30), param={"n_epoch": 20,
                                                            "n_hidden": 40})

    # network.show_neurons()
    plot(network)
    #
    # spam = network.neurons["output"][0].weights_hebbian
    # # network.train(6)
    # eggs = network.neurons["output"][0].weights_hebbian
    # print("\nspam ", spam, "\neggs", eggs)
    # for i in network.neurons["input"]:
    #     print(i.current)


if __name__ == "__main__":
    main()

# Plotting:
# x, y = np.meshgrid(range(data.shape[0]), range(data.shape[1]))
# z = data
# c = ax.contourf(x, y, z, n_levels, cmap='viridis')
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(c, cax=cax, ticks=y_ticks) aaa
