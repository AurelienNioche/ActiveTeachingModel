import numpy as np


class PowerLaw:

    def __init__(self, n_item, param):

        self.param = param

        self.seen = np.zeros(n_item, dtype=bool)
        self.timestamps = np.zeros(n_item, dtype=object)
        self.timestamps[:] = [[] for _ in range(n_item)]

    def p(self, item, now):

        a, d = self.param

        rep = np.asarray(self.timestamps[item])
        n = len(rep)
        if n == 0:
            return 0

        delta = now - rep
        if np.min(delta) == 0:
            return 1

        with np.errstate(over='ignore'):
            m = np.sum(a*np.power(delta, -d))
        p = m / (1+m)
        return p

    def update(self, item, timestamp):

        self.seen[item] = True
        self.timestamps[item].append(timestamp)