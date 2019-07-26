import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from behavior.data_structure import Task
from learner.generic import Learner

np.seterr(all='raise')


class NetworkParam:
    """
    # Architecture
    :param n_neurons: int total number of neurons. "N" in the original article.
    :param p: int number of memories.

    # Activation Function
    :param tau: float decay time.

    # Gain Function
    :param theta: gain function threshold.
    :param gamma: float gain func exponent. Always sub-linear, then value < 1.

    # Hebbian Rule
    :param kappa: excitation parameter.
    :param f: sparsity of bit probability.

    # Inhibition Parameter Calculation
    :param phi_min: float inhibition parameter minimum value.
    :param phi_max: float inhibition parameter maximum value.
    :param tau_0: oscillation time.

    # Short Term Association of Items
    :param j_forward: forward contiguity.
    :param j_backward: backward contiguity.

    # Time-related Attributes
    :param t_tot: total time when working on continuous time.
    :param dt: float integration time step.

    # Not Yet Classified
    :param xi_0: noise variance.
    :param r_threshold: recall threshold.
    :param n_trials: number of trials, which corresponds to the number of
        simulated networks.
    :param r_ini: initial rate. All neurons belonging to memory \mu are
    initialized to this value. Others are set to 0.


    Gives the parameters to the network. Default values as listed in Recanatesi
    (2015).
    """
    def __init__(self,
                 # Architecture
                 n_epoch=3,
                 n_neurons=100000,
                 p=16,
                 # Activation #############
                 tau=0.01,
                 # Gain ###################
                 theta=0,
                 gamma=0.4,
                 # Hebbian rule ###########
                 kappa=13000,
                 f=0.01,
                 # Inhibition #############
                 phi_min=0.7,
                 phi_max=1.06,
                 tau_0=1,
                 # Short term association #
                 j_forward=1500,
                 j_backward=400,
                 # Time ###################
                 t_tot=450,
                 dt=0.001,
                 # Not classified #########
                 xi_0=65,
                 r_threshold=15,
                 n_trials=10000,
                 r_ini=1,
                 verbose=False):

        # Architecture
        self.n_epoch = n_epoch
        self.n_neurons = n_neurons
        self.p = p

        # Activation function
        self.tau = tau

        # Gain function
        self.theta = theta
        self.gamma = gamma

        # Hebbian rule
        self.kappa = kappa
        self.f = f

        # Inhibition
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.tau_0 = tau_0

        # Short term association
        self.j_forward = j_forward
        self.j_backward = j_backward

        # Time
        self.t_tot = t_tot
        self.dt = dt

        # Not yet classified
        self.xi_0 = xi_0
        self.r_threshold = r_threshold
        self.n_trials = n_trials
        self.r_ini = r_ini
        self.verbose = verbose


class Network:

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

        self.connectivity = \
            np.random.choice([0, 1], p=[self.pr.f, 1 - self.pr.f],
                             size=(self.pr.p, self.pr.n_neurons))

        # self.connectivity = np.zeros((self.pr.p, self.pr.n_neurons))
        self.weights = np.zeros((self.pr.n_neurons, self.pr.n_neurons))
        self.activation = np.zeros(self.pr.n_neurons)

        self.phi = None
        self.t_tot_discrete = None

        self.t_tot_discrete = self.pr.t_tot / self.pr.dt

        self._initialize()

    def _initialize(self):
        """
        Performs the following initial operations:
        * Update the inhibition parameter for time step 0.
        * Gives a random seeded pattern as the initial activation vector.
        * Calculates the total discrete time from the total continuous time
        """
        self.update_phi(0)

        self.update_weights()

        self._present_pattern()

    def update_phi(self, t):
        self.phi = np.sin(2 * np.pi * self.pr.tau_0 * t + (np.pi / 2))\
                   * np.cos(np.pi * t + (np.pi / 2))

    def update_weights(self):
        # try:
        for i in range(self.weights.shape[0]):  # P
            print("i", i)
            for j in range(self.weights.shape[1]):  # N
                print("j", j)

                sum_ = 0
                for mu in range(self.pr.p):
                    sum_ += \
                        (self.connectivity[mu, i] - self.pr.f) \
                        * (self.connectivity[mu, j] - self.pr.f) \
                        - self.phi

                self.weights[i, j] = \
                    (self.pr.kappa / self.pr.n_neurons) * sum_
        # except:
        #     pass
        # print(self.weights)

    def _present_pattern(self):
        """
        """
        # np_question = np.array([question])
        # pattern = (((np_question[:, None]
        #             & (1 << np.arange(8))) > 0).astype(int)) * self.pr.r_ini
        # assert np.amax(pattern) == self.pr.r_ini
        # pattern = np.resize(pattern, self.activation.shape)
        # return pattern
        i = np.random.choice(np.arange(self.pr.p))
        self.activation[:] = self.connectivity[i, :]


def main():

    np.random.seed(123)

    network = Network(tk=Task(t_max=100, n_item=30),
                      param={"n_neurons": 10000})


if __name__ == main():
    main()
