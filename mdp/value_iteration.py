import math as math

import numpy as np

from mdp.mdp import MDP, _printVerbosity, \
    _MSG_STOP_EPSILON_OPTIMAL_POLICY, _MSG_STOP_MAX_ITER, util as _util


class RelativeValueIteration(MDP):

    """A MDP solved using the relative value iteration algorithm.

    Arguments
    ---------
    transitions : array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    epsilon : float, optional
        Stopping criterion. See the documentation for the ``MDP`` class for
        details. Default: 0.01.
    max_iter : int, optional
        Maximum number of iterations. See the documentation for the ``MDP``
        class for details. Default: 1000.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.

    Data Attributes
    ---------------
    self.policy : tuple
        epsilon-optimal policy
    self.average_reward  : tuple
        average reward of the optimal policy
    self.time : float
        used CPU time

    Notes
    -----
    In verbose mode, at each iteration, displays the span of U variation
    and the condition which stopped iterations : epsilon-optimum policy found
    or maximum number of iterations reached.

    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> rvi = RelativeValueIteration(P, R)
    >>> rvi.run()
    >>> rvi.average_reward
    3.2399999999999993
    >>> rvi.policy
    (0, 0, 0)
    >>> rvi.iter
    4

    >>> import mdptoolbox
    >>> import numpy as np
    >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> rvi = RelativeValueIteration(P, R)
    >>> rvi.run()
    >>> expected = (10.0, 3.885235246411831)
    >>> all(expected[k] - rvi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> rvi.average_reward
    3.8852352464118312
    >>> rvi.policy
    (1, 0)
    >>> rvi.iter
    29

    """

    def __init__(self, transitions, reward, epsilon=0.01, max_iter=1000,
                 skip_check=False):
        # Initialise a relative value iteration MDP.

        MDP.__init__(self,  transitions=transitions,
                     reward=reward, epsilon=epsilon, max_iter=max_iter,
                     skip_check=skip_check)

        self.epsilon = epsilon
        self.discount = 1

        self.V = np.zeros(self.S)
        self.gain = 0  # self.U[self.S]

        self.average_reward = None

    def run(self):
        # Run the relative value iteration algorithm.

        self._startRun()

        while True:

            self.iter += 1

            self.policy, Vnext = self._bellmanOperator()
            Vnext = Vnext - self.gain

            variation = _util.getSpan(Vnext - self.V)

            if self.verbose:
                _printVerbosity(self.iter, variation)

            if variation < self.epsilon:
                self.average_reward = self.gain + (Vnext - self.V).min()
                if self.verbose:
                    print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
                break
            elif self.iter == self.max_iter:
                self.average_reward = self.gain + (Vnext - self.V).min()
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                break

            self.V = Vnext
            self.gain = float(self.V[self.S - 1])

        self._endRun()


class ValueIteration(MDP):

    """A discounted MDP solved using the value iteration algorithm.

    Description
    -----------
    ValueIteration applies the value iteration algorithm to solve a
    discounted MDP. The algorithm consists of solving Bellman's equation
    iteratively.
    Iteration is stopped when an epsilon-optimal policy is found or after a
    specified number (``max_iter``) of iterations.
    This function uses verbose and silent modes. In verbose mode, the function
    displays the variation of ``V`` (the value function) for each iteration and
    the condition which stopped the iteration: epsilon-policy found or maximum
    number of iterations reached.

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
    epsilon : float, optional
        Stopping criterion. See the documentation for the ``MDP`` class for
        details.  Default: 0.01.
    max_iter : int, optional
        Maximum number of iterations. If the value given is greater than a
        computed bound, a warning informs that the computed bound will be used
        instead. By default, if ``discount`` is not equal to 1, a bound for
        ``max_iter`` is computed, otherwise ``max_iter`` = 1000. See the
        documentation for the ``MDP`` class for further details.
    initial_value : array, optional
        The starting value function. Default: a vector of zeros.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.

    Data Attributes
    ---------------
    self.V : np.array
        The optimal value function.
    self.policy : np.array
        The optimal policy function. Each element is an integer corresponding
        to an action which maximises the value function in that state.
    self.iter : int
        The number of iterations taken to complete the computation.
    self.time : float
        The amount of CPU time used to run the algorithm.

    Methods
    -------
    run()
        Do the algorithm iteration.
    setSilent()
        Sets the instance to silent mode.
    setVerbose()
        Sets the instance to verbose mode.

    Notes
    -----
    In verbose mode, at each iteration, displays the variation of V
    and the condition which stopped iterations: epsilon-optimum policy found
    or maximum number of iterations reached.

    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P, R = mdptoolbox.example.forest()
    >>> vi = ValueIteration(P, R, 0.96)
    >>> vi.verbose
    False
    >>> vi.run()
    >>> expected = (5.93215488, 9.38815488, 13.38815488)
    >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vi.policy
    (0, 0, 0)
    >>> vi.iter
    4

    >>> import mdptoolbox
    >>> import numpy as np
    >>> P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> vi = ValueIteration(P, R, 0.9)
    >>> vi.run()
    >>> expected = (40.048625392716815, 33.65371175967546)
    >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vi.policy
    (1, 0)
    >>> vi.iter
    26

    >>> import mdptoolbox
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix as sparse
    >>> P = [None] * 2
    >>> P[0] = sparse([[0.5, 0.5],[0.8, 0.2]])
    >>> P[1] = sparse([[0, 1],[0.1, 0.9]])
    >>> R = np.array([[5, 10], [-1, 2]])
    >>> vi = ValueIteration(P, R, 0.9)
    >>> vi.run()
    >>> expected = (40.048625392716815, 33.65371175967546)
    >>> all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vi.policy
    (1, 0)

    """

    def __init__(self, transitions, reward, discount=0.9, epsilon=0.01,
                 max_iter=1000, initial_value=0, skip_check=False):
        # Initialise a value iteration MDP.

        MDP.__init__(self, transitions=transitions,
                     reward=reward, discount=discount,
                     epsilon=epsilon, max_iter=max_iter,
                     skip_check=skip_check)

        # initialization of optional arguments
        if initial_value == 0:
            self.V = np.zeros(self.S)
        else:
            assert len(initial_value) == self.S, "The initial value must be " \
                "a vector of length S."
            self.V = np.array(initial_value).reshape(self.S)
        if self.discount < 1:
            # compute a bound for the number of iterations and update the
            # stored value of self.max_iter
            self._boundIter(epsilon)
            # computation of threshold of variation for V for an epsilon-
            # optimal policy
            self.thresh = epsilon * (1 - self.discount) / self.discount
        else:  # discount == 1
            # threshold of variation for V for an epsilon-optimal policy
            self.thresh = epsilon

    def _boundIter(self, epsilon):
        # Compute a bound for the number of iterations.
        #
        # for the value iteration
        # algorithm to find an epsilon-optimal policy with use of span for the
        # stopping criterion
        #
        # Arguments -----------------------------------------------------------
        # Let S = number of states, A = number of actions
        #    epsilon   = |V - V*| < epsilon,  upper than 0,
        #        optional (default : 0.01)
        # Evaluation ----------------------------------------------------------
        #    max_iter  = bound of the number of iterations for the value
        #    iteration algorithm to find an epsilon-optimal policy with use of
        #    span for the stopping criterion
        #    cpu_time  = used CPU time
        #
        # See Markov Decision Processes, M. L. Puterman,
        # Wiley-Interscience Publication, 1994
        # p 202, Theorem 6.6.6
        # k =    max     [1 - S min[ P(j|s,a), p(j|s',a')] ]
        #     s,a,s',a'       j
        # k = 0
        h = np.zeros(self.S)

        for ss in range(self.S):
            PP = np.zeros((self.A, self.S))
            for aa in range(self.A):
                try:
                    PP[aa] = self.P[aa][:, ss]
                except ValueError:
                    PP[aa] = self.P[aa][:, ss].todense().A1
            # minimum of the entire array.
            h[ss] = PP.min()

        k = 1 - h.sum()
        Vprev = self.V
        null, value = self._bellmanOperator()
        # p 201, Proposition 6.6.5
        span = _util.getSpan(value - Vprev)
        max_iter = (math.log((epsilon * (1 - self.discount) / self.discount) /
                             span) / math.log(self.discount * k))
        # self.V = Vprev

        self.max_iter = int(math.ceil(max_iter))

    def run(self):
        # Run the value iteration algorithm.
        self._startRun()

        while True:
            self.iter += 1

            Vprev = self.V.copy()

            # Bellman Operator: compute policy and value functions
            self.policy, self.V = self._bellmanOperator()

            # The values, based on Q. For the function "max()": the option
            # "axis" means the axis along which to operate. In this case it
            # finds the maximum of the the rows. (Operates along the columns?)
            variation = _util.getSpan(self.V - Vprev)

            if self.verbose:
                _printVerbosity(self.iter, variation)

            if variation < self.thresh:
                if self.verbose:
                    print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
                break
            elif self.iter == self.max_iter:
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                break

        self._endRun()


class ValueIterationGS(ValueIteration):

    """
    A discounted MDP solved using the value iteration Gauss-Seidel algorithm.

    Parameters
    ----------
    transitions : array
        Transition probability matrices. See the documentation for the ``MDP``
        class for details.
    reward : array
        Reward matrices or vectors. See the documentation for the ``MDP`` class
        for details.
    discount : float
        Discount factor. See the documentation for the ``MDP`` class for
        details.
    epsilon : float, optional
        Stopping criterion. See the documentation for the ``MDP`` class for
        details. Default: 0.01.
    max_iter : int, optional
        Maximum number of iterations. See the documentation for the ``MDP``
        and ``ValueIteration`` classes for details. Default: computed.
    initial_value : array, optional
        The starting value function. Default: a vector of zeros.
    skip_check : bool
        By default we run a check on the ``transitions`` and ``rewards``
        arguments to make sure they describe a valid MDP. You can set this
        argument to True in order to skip this check.

    Data Attributes
    --------------
    self.policy : np.array
        epsilon-optimal policy
    self.iter : int
        number of done iterations
    self.time : float
        used CPU time

    Notes
    -----
    In verbose mode, at each iteration, displays the variation of V
    and the condition which stopped iterations: epsilon-optimum policy found
    or maximum number of iterations reached.

    Examples
    --------
    >>> import mdptoolbox.example, numpy as np
    >>> P, R = mdptoolbox.example.forest()
    >>> vigs = ValueIterationGS(P, R, 0.9)
    >>> vigs.run()
    >>> expected = (25.5833879767579, 28.830654635546928, 32.83065463554693)
    >>> all(expected[k] - vigs.V[k] < 1e-12 for k in range(len(expected)))
    True
    >>> vigs.policy
    (0, 0, 0)

    """

    def __init__(self, transitions, reward, discount, epsilon=0.01,
                 max_iter=10, initial_value=0, skip_check=False):
        # Initialise a value iteration Gauss-Seidel MDP.

        MDP.__init__(self, transitions=transitions,
                     reward=reward, discount=discount,
                     epsilon=epsilon, max_iter=max_iter,
                     skip_check=skip_check)

        # initialization of optional arguments
        if initial_value == 0:
            self.V = np.zeros(self.S)
        else:
            if len(initial_value) != self.S:
                raise ValueError("The initial value must be a vector of "
                                 "length S.")
            else:
                try:
                    self.V = initial_value.reshape(self.S)
                except AttributeError:
                    self.V = np.array(initial_value)
                except:
                    raise
        if self.discount < 1:
            # compute a bound for the number of iterations and update the
            # stored value of self.max_iter
            self._boundIter(epsilon)
            # computation of threshold of variation for V for an epsilon-
            # optimal policy
            self.thresh = epsilon * (1 - self.discount) / self.discount
        else:  # discount == 1
            # threshold of variation for V for an epsilon-optimal policy
            self.thresh = epsilon

    def run(self):
        # Run the value iteration Gauss-Seidel algorithm.

        self._startRun()

        while True:
            self.iter += 1

            Vprev = self.V.copy()

            for s in range(self.S):
                Q = [float(self.R[a][s] +
                           self.discount * self.P[a][s, :].dot(self.V))
                     for a in range(self.A)]

                self.V[s] = max(Q)

            variation = _util.getSpan(self.V - Vprev)

            if self.verbose:
                _printVerbosity(self.iter, variation)

            if variation < self.thresh:
                if self.verbose:
                    print(_MSG_STOP_EPSILON_OPTIMAL_POLICY)
                break
            elif self.iter == self.max_iter:
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                break

        self.policy = []
        for s in range(self.S):
            Q = np.zeros(self.A)
            for a in range(self.A):
                Q[a] = (self.R[a][s] +
                        self.discount * self.P[a][s, :].dot(self.V))

            self.V[s] = Q.max()
            self.policy.append(int(Q.argmax()))

        self._endRun()
