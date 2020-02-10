import numpy as np

from . generic import Learner


class QLearner(Learner):

    version = 2.0
    bounds = ('alpha', 0, 1), ('tau', 0.002, 0.5)

    def __init__(self, param, n_item, **kwargs):

        super().__init__(**kwargs)

        self.alpha = None
        self.tau = None

        self.set_cognitive_parameters(param)

        self.q = np.zeros(n_item)

    def _temporal_difference(self, v, obs=1):

        return v + self.alpha * (obs - v)

    def _softmax(self, x):

        try:
            return 1 / (1+np.exp(1-(x/self.tau)))

        except (Warning, RuntimeWarning, FloatingPointError) as w:
            raise Exception(f'{w} [x={x}, temp={self.tau}]')

    def p_recall(self, item, time=None):

        return self._softmax(self.q[item])

    def learn(self, item, time=None, success=None):

        self.q[item] = self._temporal_difference(v=self.q[item])

    def unlearn(self):
        raise NotImplementedError
