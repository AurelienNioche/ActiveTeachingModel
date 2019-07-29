# import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

from behavior.data_structure import Task

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
                 # Architecture ###########
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
                 phi_min=0.70,
                 phi_max=1.06,
                 tau_0=1,
                 # Short term association #
                 j_forward=1500,
                 j_backward=400,
                 # Time ###################
                 t_tot=450,
                 dt=0.001,
                 # Not yet classified #####
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
        assert phi_max > phi_min
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

    def __init__(self, param, verbose=False):

        super().__init__()

        # self.tk = tk
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
            np.random.choice([0, 1], p=[1 - self.pr.f, self.pr.f],
                             size=(self.pr.p, self.pr.n_neurons))

        self.weights = np.zeros((self.pr.n_neurons, self.pr.n_neurons))
        self.activation = np.zeros(self.pr.n_neurons)

        self.phi = None
        self.t_tot_discrete = None

        self.t_tot_discrete = int(self.pr.t_tot / self.pr.dt)
        self.noise_sigma = math.sqrt(self.pr.xi_0)  # Standard deviation noise

        self.phi_history = np.zeros(self.t_tot_discrete)  # Plotting phi
        self.last_weights = np.zeros_like(self.weights)
        self.weights_history = None
        self._initialize()

    def _update_phi(self, t):
        """
        Phi is a sinusoid function related to neuron inhibition.
        It follows the general sine wave function:
            f(x) = amplitude * (2 * pi * frequency * time)
        In order to adapt to the phi_min and phi_max parameters the amplitude
        and an additive shift term have to change.
        Frequency in discrete time has to be adapted from the given tau_0
        period in continuous time.

        :param t: int discrete time step.
        """
        # phi = np.sin(2 * np.pi * self.pr.tau_0 * t + (np.pi / 2))\
        #            * np.cos(np.pi * t + (np.pi / 2))

        amplitude = (self.pr.phi_max - self.pr.phi_min) / 2
        frequency = (1 / self.pr.tau_0) * self.pr.dt
        # phase = np.random.choice()
        shift = self.pr.phi_min + amplitude  # Moves the wave in the y-axis

        self.phi = amplitude * np.sin(2 * np.pi * t * frequency) + shift

        assert self.pr.phi_max >= self.phi >= self.pr.phi_min

        self.phi_history[t] = self.phi

    def update_weights(self):
        print("Updating weights...", end=' ', flush=True)
        for i in range(self.weights.shape[0]):  # P
            for j in range(self.weights.shape[1]):  # N

                sum_ = 0
                for mu in range(self.pr.p):
                    sum_ += \
                        (self.connectivity[mu, i] - self.pr.f) \
                        * (self.connectivity[mu, j] - self.pr.f) \
                        - self.phi

                self.weights[i, j] = \
                    (self.pr.kappa / self.pr.n_neurons) * sum_
        print("Done!")
        # print(np.amax(self.weights - self.last_weights))

    def _present_pattern(self):
        # np_question = np.array([question])
        # pattern = (((np_question[:, None]
        #             & (1 << np.arange(8))) > 0).astype(int)) * self.pr.r_ini
        # assert np.amax(pattern) == self.pr.r_ini
        # pattern = np.resize(pattern, self.activation.shape)
        # return pattern
        chosen_pattern_number = np.random.choice(np.arange(self.pr.p))
        print(f"Chosen pattern number {chosen_pattern_number} of {self.pr.p}")

        self.activation[:] = self.connectivity[chosen_pattern_number, :]
        print('\nactivation\n', self.activation)

    def _initialize(self):
        """
        Performs the following initial operations:
        * Update the inhibition parameter for time step 0.
        * Updates weights according using the previous connectivity matrix
        * Gives a random seeded pattern as the initial activation vector.
        * Updates the network for the total discrete time steps.
        """
        self._update_phi(0)

        self.update_weights()

        self._present_pattern()

        self.simulate()

    def g(self, current):

        if current + self.pr.theta > 0:
            gain = (current + self.pr.theta) ** self.pr.gamma
        else:
            gain = 0

        return gain

    def gaussian_noise(self):
        return np.random.normal(loc=0, scale=self.noise_sigma)

    def _update_activation(self):

        for i in range(self.activation.shape[0]):

            try:

                current = self.activation[i]

                sum_ = 0
                for j in range(self.pr.n_neurons):
                    sum_ += self.weights[i, j] * self.g(self.activation[j])

                multiplier = \
                    - current + sum_ + self.gaussian_noise()

                self.activation[i] *= multiplier * (self.pr.tau * (1/self.pr.dt))

            except FloatingPointError as e:
                print("current", current)
                print("sum", sum_)
                print("i", i)
                print("multiplier", multiplier)
                raise e

    def simulate(self):
        print(f"Simulating for {self.t_tot_discrete} time steps...\n")
        for t in range(self.t_tot_discrete):
            self._update_phi(t)
            print("\nphi\n", self.phi)
            self.update_weights()
            print("\nweights \n", self.weights)
            # print(np.sum(self.weights))
            # print(self.phi)
            self._update_activation()
            print("\nactivation \n", self.activation)

            break


def plot_phi(network):
    data = network.phi_history
    time = np.arange(0, network.phi_history.size, 1)

    plt.plot(time, data)
    plt.title("Inhibitory oscillations")
    plt.xlabel("Time")
    plt.ylabel("Phi")

    plt.show()

    print(np.amax(network.phi_history))
    print(np.amin(network.phi_history))


def plot_weights(network, time):
    data = network.weights

    fig, ax = plt.subplots()
    im = ax.imshow(data)

    plt.title(f"Weights matrix (t = {time})")
    plt.xlabel("Weights")
    plt.ylabel("Weights")

    fig.tight_layout()

    plt.show()


def main():

    np.random.seed(123)

    network = Network(param={"n_neurons": 10**5,
                             'f': 0.1,
                      'p': 16, "kappa": 13*10**3, "t_tot": 450})

    # plot_phi(network)


if __name__ == main():
    main()
