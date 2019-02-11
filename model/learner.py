import numpy as np

from utils.functions import temporal_difference, softmax
from task import exercise


debug = False


class QLearner:

    def __init__(self, alpha=0.1, tau=0.05):

        self.q = np.zeros((exercise.n, exercise.n))
        self.alpha = alpha
        self.tau = tau

    def decide(self, question):

        p = softmax(x=self.q[question, :], temp=self.tau)
        reply = np.random.choice(np.arange(exercise.n), p=p)

        if debug:
            print(f'Question is: {question}')
            print(f'P values are: {[f"{p_i:.02f}" for p_i in p]}')
            print(f'Reply is {reply}')

        return reply

    def learn(self, question, reply, correct_answer):

        success = reply == correct_answer

        old_q_value = self.q[question, reply]
        new_q_value = temporal_difference(v=old_q_value, obs=success, alpha=self.alpha)

        self.q[question, reply] = new_q_value

        if not success:
            self.q[question, correct_answer] = temporal_difference(v=self.q[question, correct_answer], obs=1, alpha=self.alpha)

        if debug:
            print(f'Old q value is {old_q_value}; New q value is {new_q_value}')


class ActRLearner:

    def __init__(self, d=0.5, p=0.01, theta=0.5, sigma=0.01):

        # Noted 'l' in the paper (to confirm)
        self.n_chunk = exercise.n

        # Noted 'n' in the paper (number of source of activation)
        self.n_content = exercise.n

        # Strength (ji-indexed): Sji is the existing strength of association from element j to chunk i
        self.s = np.zeros((self.n_content, self.n_chunk))

        # Activation of chunks (i-indexed)
        self.a = np.zeros(self.n_chunk)

        # Base level activations (i-indexed)
        self.b = np.zeros(self.n_chunk)

        # Wj source activation of element j currently attended to
        self.w = np.zeros(self.n_content)

        #
        self.t = np.zeros(self.n_content)

        # The goodness of the match Mi of a chunk i
        self.m = np.zeros(self.n_chunk)

        # The probability of a chunk being above some retrieval threshold τ is
        self.prob = np.zeros(self.n_chunk)

        # The probability of retrieving a chunk is
        self.probability_retrieving_chunk = np.zeros(self.n_chunk)

        # Decay parameter
        self._d = d

        # Mismatch parameter
        self._p = p

        self._theta = theta

        self._sigma = sigma

        # Scale constant
        self._s = 0

        self._t = 0

    def set_w(self):

        """
        If there are n sources of activation, the Wj are set to l/n.
        :return:
        """

        self.w[:] = self.n_chunk / self.n_content

    def set_s(self):

        """S is a scale constant, which by default is set to the log of the total number of chunks"""
        self._s = np.log(self.n_chunk)

    def set_t(self):

        self._t = (6**(1/2) * self._sigma) / np.pi

    def third(self, i):

        """The activation of a chunk is the sum of its base-level activation
        and the activations it receives from elements attended to. """

        self.a[i] = self.b[i] + np.sum([self.w[j] * self.s[j, i] for j in range(self.n_content)])

    def fourth(self, i):

        """The base-level activation measures how much time has elapsed since the jth use:"""

        self.b[i] = np.log(np.sum([self.t[j]**(-self._d) for j in range(self.n_content)]))

    def fifth(self, j, i):

        self.s[j, i] = self._s - np.log(self.p(i, j))

    def sixth(self, i):

        self.m[i] = self.a[i] - self._p

    def seventh(self, i):

        """The probability of a chunk being above some retrieval threshold τ is"""

        self.prob[i] = 1 / (1 + np.exp((self.m[i] - self._theta)/2))

    def eighth(self, i):

        """The probability of retrieving a chunk is"""
        self.probability_retrieving_chunk[i] = np.exp(self.m[i] / self._t)

    @classmethod
    def p(cls, i, j):

        """P(i | j) is the probability that chunk i will be needed when j appears in the context."""

        return 0.5

    def decide(self, question):

        pass
        # p = softmax(x=self.q[question, :], temp=self.tau)
        # reply = np.random.choice(np.arange(exercise.n), p=p)
        #
        # if debug:
        #     print(f'Question is: {question}')
        #     print(f'P values are: {[f"{p_i:.02f}" for p_i in p]}')
        #     print(f'Reply is {reply}')
        #
        # return reply

    def learn(self, question, reply, correct_answer):

        pass

        # success = reply == correct_answer
        #
        # old_q_value = self.q[question, reply]
        # new_q_value = temporal_difference(v=old_q_value, obs=success, alpha=self.alpha)
        #
        # self.q[question, reply] = new_q_value
        #
        # if not success:
        #     self.q[question, correct_answer] = temporal_difference(v=self.q[question, correct_answer], obs=1, alpha=self.alpha)
        #
        # if debug:
        #     print(f'Old q value is {old_q_value}; New q value is {new_q_value}')
