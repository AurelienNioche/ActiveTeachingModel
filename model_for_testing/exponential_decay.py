import numpy as np


class ExponentialDecay:

    def __init__(self, n_item, param):
        self.n_item = n_item
        self.param = param

        self.n_pres = np.zeros(n_item, dtype=int)
        self.last_pres = np.zeros(n_item, dtype=int)

    def p(self, item, now):
        fr = self.param[0] * (1-self.param[1])**(self.n_pres[item]-1)
        delta = now - self.last_pres[item]
        return np.exp(-fr * delta)

    def update(self, item, timestamp):
        self.n_pres[item] += 1
        self.last_pres[item] = timestamp