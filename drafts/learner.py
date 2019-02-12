# def fifth(self, j, i):
    #
    #     self.s[j, i] = self._s - np.log(self.p(i, j))
    #
    # def sixth(self, i):
    #
    #     self.m[i] = self.a[i] - self._p

    # def probability_of_retrieval_equation(self, a):
    #
    #     """The probability of a chunk being above some retrieval threshold τ is"""
    #
    #     # self.prob[i] = 1 / (1 + np.exp((self.m[i] - self._theta)/2))
    #     return \
    #         1 / (1 + np.exp(
    #                     - (a - self._theta) / (cmath.sqrt(2) * self._s)
    #                 )
    #              )
    #
    # def probability_to_win_competition(self, a, competitors):
    #
    #     """The probability that the chunk wins the competition.
    #     ks is the number of chunks with an activation above the retrieval threshold"""
    #
    #     # self.p_win[i] = np.exp(self.m[i] / self._t)
    #     return \
    #         np.exp(
    #             a / (cmath.sqrt(2) * self._s)) / \
    #         np.sum([np.exp(
    #             self.activation_function(k) / (cmath.sqrt(2) * self._s)
    #         ) for k in competitors])

    # @classmethod
    # def p(cls, i, j):
    #
    #     """P(i | j) is the probability that chunk i will be needed when j appears in the context."""
    #
    #     return 0.5


# def activation_function(self, i):
#     """The activation of a chunk is the sum of its base-level activation and some noise
#     IN FUTURE: ...and the activations it receives from elements attended to. """
#
#     # self.a[i] = self.b[i]  # + noise  # np.sum([self.w[j] * self.s[j, i] for j in range(self.n_content)])
#     # sum_k MP_k *  Sim_kl  // mismatch penalty * similarity value


# # Noted 'n' in the paper (number of source of activation)
# self.n_content = exercise.n

# Strength (ji-indexed): Sji is the existing strength of association from element j to chunk i
# self.s = np.zeros((self.n_content, self.n_chunk))
#
# self.chunks = np.zeros(())

# Wj source activation of element j currently attended to
# self.w = np.zeros(self.n_content)

# The goodness of the match Mi of a chunk i
# self.m = np.zeros(self.n_chunk)

# Mismatch parameter
# self._p = p

# self._sigma = sigma

# self._t = 0

# def set_w(self, j, n, w=1):
#
#     """
#     The Wj ’s reflect the attentional weighting of the elements that are part of the current goal,
#     and are set to W/n where n is the number of elements in the current goal, and W is a global ACT-R parameter
#     that is by default set to 1.
#     """
#
#     self.w[j] = w / n

# def set_w(self):
#
#     """
#     If there are n sources of activation, the Wj are set to l/n.
#     :return:
#     """
#
#     self.w[:] = self.n_chunk / self.n_content

# def set_s(self):
#
#     """S is a scale constant, which by default is set to the log of the total number of chunks"""
#     self._s = np.log(self.n_chunk)

# def set_t(self):
#
#     self._t = (6**(1/2) * self._sigma) / np.pi