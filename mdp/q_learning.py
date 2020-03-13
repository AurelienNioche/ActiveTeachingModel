import math as _math
import time as time

import numpy as np

from mdp.mdp import MDP, _computeDimensions, util as _util


class QLearning(MDP):

    """A discounted MDP solved using the Q learning algorithm.

    Parameters
    ----------
    transitions : np.array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : np.array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    discount : float
        Discount factor. See the documentation for the ``MDP`` class for
        details.
    n_iter : int, optional
        Number of iterations to execute. This is ignored unless it is an
        integer greater than the default value. Defaut: 10,000.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.

    Data Attributes
    ---------------
    Q : np.array
        learned Q matrix (SxA)
    V : tuple
        learned value function (S).
    policy : tuple
        learned optimal policy (S).
    mean_discrepancy : np.array
        Vector of V discrepancy mean over 100 iterations. Then the length of
        this vector for the default value of N is 100 (N/100).

    Examples
    ---------
    >>> # These examples are reproducible only if random seed is set to 0 in
    >>> # both the random and numpy.random modules.
    >>> import numpy as np
    >>> import mdptoolbox, mdptoolbox.example
    >>> np.random.seed(0)
    >>> P, R = mdptoolbox.example.forest()
    >>> ql = QLearning(P, R, 0.96)
    >>> ql.run()
    >>> ql.Q
    array([[ 11.198909  ,  10.34652034],
           [ 10.74229967,  11.74105792],
           [  2.86980001,  12.25973286]])
    >>> expected = (11.198908998901134, 11.741057920409865, 12.259732864170232)
    >>> all(expected[k] - ql.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> ql.policy
    (0, 1, 1)

    >>> import mdptoolbox
    >>> import numpy as np
    >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> np.random.seed(0)
    >>> ql = QLearning(P, R, 0.9)
    >>> ql.run()
    >>> ql.Q
    array([[ 33.33010866,  40.82109565],
           [ 34.37431041,  29.67236845]])
    >>> expected = (40.82109564847122, 34.37431040682546)
    >>> all(expected[k] - ql.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> ql.policy
    (1, 0)

    """

    def __init__(self, transitions, reward, discount, n_iter=10000,
                 skip_check=False):
        # Initialise a Q-learning MDP.

        # The following check won't be done in MDP()'s initialisation, so let's
        # do it here
        self.max_iter = int(n_iter)
        assert self.max_iter >= 10000, "'n_iter' should be greater than 10000."

        # Initialise the base class
        MDP.__init__(self, transitions=transitions, reward=reward,
                     discount=discount,
                     skip_check=skip_check)

        self.R = reward

        self.discount = discount

        # Initialisations
        self.Q = np.zeros((self.S, self.A))
        self.mean_discrepancy = []

    def run(self):
        # Run the Q-learning algoritm.
        discrepancy = []

        self.time = time.time()

        # initial state choice
        s = np.random.randint(0, self.S)

        for n in range(1, self.max_iter + 1):

            # Reinitialisation of trajectories every 100 transitions
            if (n % 100) == 0:
                s = np.random.randint(0, self.S)

            # Action choice : greedy with increasing probability
            # probability 1-(1/log(n+2)) can be changed
            pn = np.random.random()
            if pn < (1 - (1 / _math.log(n + 2))):
                # optimal_action = self.Q[s, :].max()
                a = self.Q[s, :].argmax()
            else:
                a = np.random.randint(0, self.A)

            # Simulating next state s_new and reward associated to <s,s_new,a>
            p_s_new = np.random.random()
            p = 0
            s_new = -1
            while (p < p_s_new) and (s_new < (self.S - 1)):
                s_new = s_new + 1
                p = p + self.P[a][s, s_new]

            try:
                r = self.R[a][s, s_new]
            except IndexError:
                try:
                    r = self.R[s, a]
                except IndexError:
                    r = self.R[s]

            # Updating the value of Q
            # Decaying update coefficient (1/sqrt(n+2)) can be changed
            delta = r + self.discount * self.Q[s_new, :].max() - self.Q[s, a]
            dQ = (1 / _math.sqrt(n + 2)) * delta
            self.Q[s, a] = self.Q[s, a] + dQ

            # current state is updated
            s = s_new

            # Computing and saving maximal values of the Q variation
            discrepancy.append(np.absolute(dQ))

            # Computing means all over maximal Q variations values
            if len(discrepancy) == 100:
                self.mean_discrepancy.append(np.mean(discrepancy))
                discrepancy = []

            # compute the value function and the policy
            self.V = self.Q.max(axis=1)
            self.policy = self.Q.argmax(axis=1)

        self._endRun()