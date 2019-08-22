# This is the modified version of carlos_ann in order to adapt it to the true
# Hopfield architecture, dropped in favor of a matrix architecture

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from behavior.data_structure import Task
from learner.generic import Learner

np.seterr(all='raise')


class Neuron:
    """
    :param input_neurons: list of connected to this neuron
    :param tau: decay time
    :param theta: gain function threshold
    :param gamma: gain func exponent. Always sub-linear, then value < 1
    :param kappa: excitation parameter
    :param phi: inhibition parameter in range(0.70, 1.06)
    :param f: sparsity of bit probability

    Modified from Recanatesi (2015) equations 1 to 4.
    """

    def __init__(self, network, tau=0.01, theta=0, gamma=0.4,
                 kappa=13000, f=0.1,
                 input_neuron_ids=None, current=0):

        self.network = network

        self.input_neuron_ids = input_neuron_ids

        self.input_currents = None
        self.weights = None
        self.weights_hebbian = None
        self.gain = None

        # Neuron dynamics
        self.current = current
        self.tau = tau
        # self.input_neurons = input_neuron_ids

        # Gain function
        self.theta = theta
        self.gamma = gamma

        # Hebbian rule
        self.kappa = kappa
        self.f = f

    def start(self):

        self._initialize_attributes()

        for i in range(0, 1):
            self.compute_gain()
            self.update_current()

    @staticmethod
    def _random_small_value():
        number = abs(np.random.normal(loc=0.5, scale=0.25))
        number = max(min(1, number), 0)
        return number

    def _initialize_attributes(self):
        self.input_currents = \
            np.array([self._random_small_value()
                      for _ in range(len(self.input_neurons))])
        self.weights = np.random.rand(1, len(self.input_neurons))
        self.current = self._random_small_value()

    def compute_gain(self):
        if self.current + self.theta > 0:
            self.gain = (self.current + self.theta) ** self.gamma
        else:
            self.gain = 0

    def update_current(self):
        self.weights_hebbian = self.kappa / 100000\
            * (np.sum(self.current * self.input_currents)
                - self.phi)
        r = self._random_small_value()

        try:
            sum_gain = self.weights_hebbian * self.gain

            self.current *= \
                (-self.current + sum_gain + r)  # / self.tau

        except FloatingPointError as e:
            print('gain:', self.gain)
            print()
            print('weights_hebbian', self.weights_hebbian)
            raise e

        # if self.role == "hidden":
        self.current = int(self.current > 0.5)

    def print_attributes(self):
        """
        :returns: multi-line string of instance attributes
        """
        print(
            '* '.join("%s: %s\n" % item for item in vars(self).items())
        )


class NetworkParam:
    """
    :param t_max: int as in the model_simulation script. None when no
        simulation script is being used with the model
    """
    def __init__(self, n_neurons=40, p=16, n_epoch=3,
                 t_max=None):
        self.n_epoch = n_epoch
        self.n_neurons = n_neurons
        self.t_max = t_max
        self.p = p


class Network(Learner):

    version = 4.0
    bounds = ('d', 0.001, 1.0), \
             ('tau', -1, 1), \
             ('s', 0.001, 1)  # TODO change

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

        # Create history array according to simulation or the lack thereof
        if self.pr.t_max is None:
            self.currents_history = np.zeros((self.pr.n_epoch,
                                              self.pr.n_neurons))
        else:
            self.currents_history = np.zeros((self.pr.t_max,
                                              self.pr.n_neurons))
            self.time_step = 0

        self.neurons = []
        self.create_neurons(n_neurons=self.pr.n_neurons)
        self.connect_everything()
        self.start_network()

        self._train(self.pr.n_epoch)

    def create_neurons(self, n_neurons):

        for j in range(n_neurons):

            neuron_inputs = []

            for i in range(self.pr.n_neurons):
                neuron_inputs.append(i)

            self.neurons = list(range(n_neurons))
            self.neurons.remove(j)
            for n in self.neurons:
                neuron_inputs.append(n)

            self._create_neuron(input_neuron_ids=neuron_inputs)

    def _create_neuron(self, input_neuron_ids=None):

        self.neurons.append(
            Neuron(network=self, input_neuron_ids=input_neuron_ids)
        )

    def connect_everything(self):

        for n in self.neurons:
            connected_neurons = []
            for k, v in connected_neurons:
                connected_neurons.append(
                    self.neurons[k][v]
                )

                n.input_neurons = connected_neurons

    def start_network(self):

        for n in self.neurons:
            n.start()

    def _train(self, n_epochs):
        """
        :param n_epochs: int in range(0, +inf). Number of epochs
        """

        if n_epochs < 0:
            raise ValueError("n_epochs not int or not in range(0, +inf)")

        for i in tqdm(range(n_epochs)):
            for neuron in self.neurons:
                neuron.compute_gain()
                neuron.update_current()
                if self.pr.t_max is None:  # Only when NOT simulating
                    self._update_hidden_currents_history(i)

    def _update_hidden_currents_history(self, time_step):
        for j, val in enumerate(self.neurons):
            self.currents_history[time_step, j] = \
                self.neurons.current

    def p_recall(self, item, time=None):
        p_recall = 1  # TODO output p_r
        return p_recall

    def _p_choice(self, item, reply, possible_replies=None,
                  time=None, time_index=None):
        """Modified from ActR"""

        success = item == reply

        p_recall = 1  # TODO output p_r

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

    def _p_correct(self, item, reply, possible_replies=None,
                   time=None, time_index=None):

        p_correct = self._p_choice(item=item, reply=item,
                                   time=time, time_index=time_index)

        correct = item == reply
        if correct:
            return p_correct

        else:
            return 1-p_correct

    def decide(self, item, possible_replies, time=None,
               time_index=None):

        # for j, val in enumerate(self.neurons["input"]):
        #     self.currents_history[time_step, j] = \
        #         self.neurons["hidden"][j].current

        self._update_input_currents(item)
        self._train(self.pr.n_epoch)

        # p_r = self.neurons["output"][0].current
        number = abs(np.random.normal(loc=0.5, scale=0.25))
        number = max(min(1, number), 0)
        p_r = number  # TODO ugly fix as we removed the output layer
        r = np.random.random()

        if p_r > r:
            reply = item
        else:
            reply = np.random.choice(possible_replies)

        self._update_hidden_currents_history(self.time_step)
        self.time_step += 1
        if self.time_step == self.pr.t_max:
            plot(self)

        if self.verbose:
            print(f't={self.t}: question {item}, reply {reply}')
        return reply

    def learn(self, item, time=None):
        pass

    def unlearn(self):
        pass


def plot(network):

    data = network.hidden_currents_history

    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # ax.set_xticks(np.arange(len(data[0, :])))
    # font_size = 40 * 8 / len(data[0, :])  # Font size 10 good for 40 neurons
    # TODO fix font size
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")  # , fontsize=font_size)

    plt.title("Hidden layer currents")

    fig.tight_layout()
    plt.show()


def main():

    np.random.seed(1234)

    network = Network(tk=Task(t_max=100, n_item=30), param={"n_epoch": 200,
                                                            "num_neurons": 10})

    # plot(network)


if __name__ == "__main__":
    main()
