import numpy as np


class ActR2008PA:

    def __init__(self, n_item, param):

        tau, s, c, a = param

        self.cst0 = np.exp(tau/s)
        self.cst1 = - c * np.exp(tau)
        self.min_s = -s
        self.min_inv_s = -1/s
        self.min_a = -a

        self.seen = np.zeros(n_item, dtype=bool)
        self.timestamps = np.zeros(n_item, dtype=object)
        self.timestamps[:] = [[] for _ in range(n_item)]

    def p(self, item, now):

        rep = np.asarray(self.timestamps[item])
        n = len(rep)
        if n == 0:
            return 0

        delta = now - rep
        if np.min(delta) == 0:
            return 1

        tk_sum = delta[0]**self.min_a
        one_minus_p = 1 + self.cst0 * tk_sum**self.min_inv_s
        with np.errstate(invalid='ignore'):
            for i in range(1, n):
                tk_sum += self.cst1 * one_minus_p**self.min_s + self.min_a
                one_minus_p = 1 + self.cst0 * tk_sum**self.min_inv_s

        p = 1/one_minus_p
        return p

    def update(self, item, timestamp):

        self.seen[item] = True
        self.timestamps[item].append(timestamp)