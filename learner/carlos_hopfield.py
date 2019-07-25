import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from behavior.data_structure import Task
from learner.generic import Learner

np.seterr(all='raise')


class NetworkParam:
    """
    :param t_max: int as in the model_simulation script. None when no
        simulation script is being used with the model
    :param tau: decay time
    :param theta: gain function threshold
    :param gamma: gain func exponent. Always sub-linear, then value < 1
    :param kappa: excitation parameter
    :param phi_min: minimum value of the inhibition parameter
    :param phi_max: maximum "
    :param f: sparsity of bit probability
    """
    def __init__(self, n_epoch=3, n_neurons=100000, p=16, tau=0.01, theta=0,
                 gamma=0.4, f=0.01, kappa=13000, phi_min=0.7, phi_max=1.06,
                 tau_0=1, t_max=None, verbose=False):

        self.n_epoch = n_epoch
        self.n_neurons = n_neurons

        # Neuron dynamics
        self.tau = tau

        # Gain function
        self.theta = theta
        self.gamma = gamma

        # Hebbian rule
        self.kappa = kappa
        self.phi = phi_min
        self.phi = phi_max
        self.f = f

        # Function parameters
        self.p = p
        self.f = f
        self.kappa = kappa
        self.tau_0 = tau_0

        self.t_max = t_max
        self.verbose = verbose


class Network(Learner):

    version = 0.1
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

        self.weights = np.zeros((self.pr.p, self.pr.n_neurons))
        self.currents = self.weights

        self.representation_memory = \
            np.random.choice([0, 1], p=[self.pr.f, 1 - self.pr.f],
                             size=(self.pr.p, self.pr.n_neurons))

        self.phi = None

        self._initialize()

    def _initialize(self):
        self.update_phi(0)
        print(self.phi)
        self.update_weights()

    def update_phi(self, t):
        self.phi = np.sin(2 * np.pi * self.pr.tau_0 * t + (np.pi / 2)) * np.cos(
            np.pi * t + (np.pi / 2))

    def update_weights(self):
        try:
            for i in range(self.weights.shape[0]):  # P
                for j in range(self.weights.shape[1]):  # N
                    self.weights[i, j] = self.pr.kappa / self.pr.n_neurons * (
                         (self.currents[i, j] - self.pr.f)
                             * (self.currents[i+1, j+1] - self.pr.f) - self.phi)
        except:
            pass
        print(self.weights)

    #####################################
    # Integration with teacher and task #
    #####################################

    def p_recall(self, item, time=None):
        p_recall = 1
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

        number = abs(np.random.normal(loc=0.5, scale=0.25))
        number = max(min(1, number), 0)
        p_r = number  # TODO ugly fix as we removed the output layer
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


def main():
    network = Network(tk=Task(t_max=100, n_item=30),
                      param={"n_neurons": 100000})


if __name__ == main():
    main()
