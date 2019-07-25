import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from sys import exit

from behavior.data_structure import Task
from learner.generic import Learner

np.seterr(all='raise')


class Neuron:
    """
    :param input_neurons: list of neuron_id connected to this neuron
    :param tau: decay time
    :param theta: gain function threshold
    :param gamma: gain func exponent. Always sub-linear, then value < 1
    :param kappa: excitation parameter
    :param phi: inhibition parameter in range(0.70, 1.06)
    :param f: sparsity of bit probability

    Modified from Recanatesi (2015) equations 1 to 4.
    """

    def __init__(self, network, neuron_id, tau=0.01, theta=0, gamma=0.4,
                 kappa=13000, f=0.1,
                 input_neuron_ids=None, current=0):

        self.network = network
        self.neuron_id = neuron_id

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
            if self.role is not "input":
                self.compute_gain()
                self.update_current()

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
            print(len(self.input_neurons))
            self.weights = np.random.rand(1, len(self.input_neurons))
            self.current = self._random_small_value()

    def compute_gain(self):
        if self.current + self.theta > 0:
            self.gain = (self.current + self.theta) ** self.gamma
        else:
            self.gain = 0

    def update_current(self):
        assert self.role is not "input"

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
    def __init__(self, n_hidden=40, n_epoch=3,
                 t_max=None):
        self.n_epoch = n_epoch
        self.n_hidden = n_hidden
        self.t_max = t_max


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
            self.hidden_currents_history = np.zeros((self.pr.n_epoch,
                                                    self.pr.n_hidden))
        else:
            self.hidden_currents_history = np.zeros((self.pr.t_max,
                                                     self.pr.n_hidden))
            self.time_step = 0

        self.create_hidden_neurons(n_hidden=self.pr.n_hidden)
        self.connect_everything()
        self.start_network()

        self._train(self.pr.n_epoch)

    def create_hidden_neurons(self, n_hidden):

        for j in range(n_hidden):

            hidden_neuron_inputs = []

            for i in range(len(self.neurons['input'])):

                hidden_neuron_inputs.append(('input', i))

            hid = list(range(n_hidden))
            hid.remove(j)
            for h in hid:
                hidden_neuron_inputs.append(('hidden', h))

            self._create_neuron(input_neuron_ids=hidden_neuron_inputs)




        self, n_neurons

        for j in range(n_neurons):

            neuron_inputs = []

            for i in range(len(self.neurons['input'])):

                hidden_neuron_inputs.append(('input', i))

            hid = list(range(n_hidden))
            hid.remove(j)
            for h in hid:
                hidden_neuron_inputs.append(('hidden', h))

            self._create_neuron(input_neuron_ids=hidden_neuron_inputs)





    def _create_neuron(self, input_neuron_ids=None, role="hidden"):

        self.neurons[role].append(
            Neuron(network=self, neuron_id=self.neuron_id,
                   role=role, input_neuron_ids=input_neuron_ids)
        )

        self.neuron_id += 1

    def connect_everything(self):

        for layer in self.neurons.keys():
            if layer == 'input':
                continue
            for n in self.neurons[layer]:

                connected_neurons = []
                for k, v in n.input_neuron_ids:

                    connected_neurons.append(
                        self.neurons[k][v]
                    )

                    n.input_neurons = connected_neurons

    def start_network(self):

        for layer in self.neurons.keys():
            for n in self.neurons[layer]:
                n.start()

    def show_neurons(self):
        for role in self.roles:
            print(f"Number of {role} neurons: ", len(self.neurons[role]))

    def _train(self, n_epochs, verbose=False):
        """
        :param n_epochs: int in range(0, +inf). Number of epochs
        """

        if n_epochs < 0:
            raise ValueError("n_epochs not int or not in range(0, +inf)")

        for i in tqdm(range(n_epochs)):
            for neuron in self.neurons["hidden"]:
                neuron.compute_gain()
                neuron.update_current()
                if self.pr.t_max is None:  # Only when NOT simulating
                    self._update_hidden_currents_history(i)

            for neuron in self.neurons["output"]:
                neuron.compute_gain()
                neuron.update_current()

        # if verbose:
        #     print(self.hidden_currents_history)

    def _update_hidden_currents_history(self, time_step):
        for j, val in enumerate(self.neurons["hidden"]):
            self.hidden_currents_history[time_step, j] = \
                self.neurons["hidden"][j].current

    def _update_input_currents(self, question):
        """
        :param question: int question index

        Question to binary vector as input neurons current.
        """
        np_question = np.array([question])
        bin_question = ((np_question[:, None]
                         & (1 << np.arange(8))) > 0).astype(int)
        for j, val in enumerate(self.neurons["input"]):
            self.neurons["input"][j].current = bin_question[0, j]

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

        # for j, val in enumerate(self.neurons["input"]):
        #     self.hidden_currents_history[time_step, j] = \
        #         self.neurons["hidden"][j].current

        self._update_input_currents(question)
        self._train(self.pr.n_epoch)

        # p_r = self.neurons["output"][0].current
        number = abs(np.random.normal(loc=0.5, scale=0.25))
        number = max(min(1, number), 0)
        p_r = number  # TODO ugly fix as we removed the output layer
        r = np.random.random()

        if p_r > r:
            reply = question
        else:
            reply = np.random.choice(possible_replies)

        self._update_hidden_currents_history(self.time_step)
        self.time_step += 1
        if self.time_step == self.pr.t_max:
            plot(self)

        if self.verbose:
            print(f't={self.t}: question {question}, reply {reply}')
        return reply

    def learn(self, question, time=None):
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
                                                            "n_hidden": 10,
                                                            "verbose": False})

    plot(network)


if __name__ == "__main__":
    main()
