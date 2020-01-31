import numpy as np


class FakeModel:

    bounds = {
        "b0": (0, 10),
        "b1": (0, 10)
    }

    def __init__(self, param, t=None, hist=None):

        if isinstance(param, dict):
            self.b0 = param['b0']
            self.b1 = param['b1']
        else:
            self.b0, self.b1 = param

        self.hist = hist
        self.t = t

    def p_recall(self, item):
        """A function to compute the probability of a positive response."""

        logit = self.b0 + item * self.b1**2
        p_obs = 1. / (1 + np.exp(-logit))

        return p_obs

    def learn(self, **kwargs):
        pass