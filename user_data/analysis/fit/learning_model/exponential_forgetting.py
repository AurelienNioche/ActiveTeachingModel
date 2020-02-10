import numpy as np

from . generic import Learner


class ExponentialForgetting(Learner):

    version = 1.0
    bounds = ('alpha', 0., 1.), ('beta', 0., 1.)

    def __init__(self, param, n_item, **kwargs):

        super().__init__(**kwargs)

        self.alpha = None
        self.beta = None

        self.set_cognitive_parameters(param)

        self.n_pres = np.zeros(n_item)
        self.last_pres = np.zeros(n_item)

        self.t = 0

    def p_recall(self, item, time=None):

        if self.n_pres[item] == 0:
            return 0

        fr = self.alpha * (1 - self.beta) ** (self.n_pres[item] - 1)

        p = np.exp(- fr * (self.t - self.last_pres[item]))
        return p

    def learn(self, item, time=None, success=None):

        self.n_pres[item] += 1
        self.last_pres[item] = self.t
        self.t += 1

    def unlearn(self):
        raise NotImplementedError


class ExponentialForgettingAsymmetric(Learner):

    version = 1.0
    bounds = ('alpha', 0., 1.), ('beta_minus', 0., 1.), \
             ('beta_plus', 0., 1.)

    def __init__(self, param, n_item, **kwargs):

        super().__init__(**kwargs)

        self.alpha = None
        self.beta_minus = None
        self.beta_plus = None

        self.set_cognitive_parameters(param)

        self.n_pres = np.zeros(n_item)
        self.n_success = np.zeros(n_item)
        self.last_pres = np.zeros(n_item)

        self.t = 0

    def p_recall(self, item, time=None):

        if self.n_pres[item] == 0:
            return 0

        fr = self.alpha \
            * (1 - self.beta_minus) ** \
            (self.n_pres[item] - self.n_success[item] - 1) \
            * (1 - self.beta_plus) ** \
            self.n_success[item]

        p = np.exp(- fr * (self.t - self.last_pres[item]))
        return p

    def learn(self, item, time=None, success=None):

        self.n_pres[item] += 1
        self.last_pres[item] = self.t
        self.t += 1

        if self.n_pres[item] > 1:
            self.n_success[item] += int(success)

    def unlearn(self):
        raise NotImplementedError
