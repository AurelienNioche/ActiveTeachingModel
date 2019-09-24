from abc import ABC

import numpy as np

from learner.generic import Learner


class QLearner(Learner, ABC):

    version = 1.0
    bounds = ('alpha', 0, 1), ('tau', 0.002, 0.5)

    def __init__(self, param, n_item, **kwargs):

        self.alpha = None
        self.tau = None

        self.set_cognitive_parameters(param)

        super().__init__(**kwargs)

        self.q = np.zeros(n_item)

        self.item = None
        self.value = None

    def _temporal_difference(self, v, obs=1):

        return v + self.alpha * (obs - v)

    def _softmax(self, x):

        try:
            return 1 / (1 + np.exp(1 - (x / self.tau)))

        except (Warning, RuntimeWarning, FloatingPointError) as w:
            raise Exception(f'{w} [x={x}, temp={self.tau}]')

    def p_recall(self, item, time=None, time_index=None):

        return self._softmax(self.q[item])

    def learn(self, item, time=None, time_index=None):

        self.item = item
        self.value = self.q[item]
        self.q[item] = self._temporal_difference(v=self.q[item])

    def unlearn(self):
        self.q[self.item] = self.value
