import numpy as np
from . generic import Learner
from scipy.special import expit
import math

EPS = np.finfo(np.float).eps


class Walsh2018(Learner):

    def __init__(self, n_item, n_iter):

        self.seen = np.zeros(n_item, dtype=bool)
        self.ts = np.full(n_iter, -1, dtype=int)
        self.hist = np.full(n_iter, -1, dtype=int)

        self.seen_item = None
        self.n_seen = 0
        self.i = 0

        self.tau = None
        self.s = None
        self.c = None
        self.x = None
        self.b = None
        self.m = None

    def p(self, item, param, now, is_item_specific):

        self.set_param(param=param)

        hist = np.asarray(self.hist)
        ts = np.asarray(self.ts)

        relevant = hist == item
        rep = ts[relevant]
        n = len(rep)
        delta = (now - rep)

        if n == 0:
            return 0
        elif np.min(delta) == 0:
            return 1
        else:

            w = delta ** -self.x
            w /= np.sum(w)

            _t_ = np.sum(w * delta)
            if n > 1:
                lag = rep[1:] - rep[:-1]
                d = self.b + self.m * np.mean(1/np.log(lag + math.e))
            else:
                d = self.b

            _m_ = n ** self.c * _t_ ** -d

            v = (-self.tau + _m_) / self.s
            p = expit(v)
            return p

    @staticmethod
    def log_lik(param, hist, success, timestamp):
        tau, s, b, m, c, x = param

        _m_ = np.zeros(len(hist))

        for item in np.unique(hist):

            relevant = hist == item
            rep = timestamp[relevant]
            n = len(rep)

            _m_item = np.zeros(n)

            _m_item[0] = - np.inf  # To adapt for xp
            _m_item[1] = (rep[1]-rep[0])**-b
            for i in range(2, n):
                delta = rep[i] - rep[:i]

                w = delta ** -x
                w /= np.sum(w)

                _t_ = np.sum(w * delta)

                lag = rep[1:i+1] - rep[:i]
                d = b + m * np.mean(1 / np.log(lag + math.e))
                _m_item[i] = i ** c * _t_ ** -d

            _m_[b] = _m_item

        with np.errstate(divide="ignore", invalid="ignore"):
            v = (-tau + _m_) / s

        p = expit(v)
        failure = np.invert(success)
        p[failure] = 1 - p[failure]
        log_lik = np.log(p + EPS)
        return log_lik.sum()

    def update(self, item, timestamp):

        self.seen[item] = True
        self.hist[self.i] = item
        self.ts[self.i] = timestamp

        self.n_seen = np.sum(self.seen)
        self.seen_item = np.flatnonzero(self.seen)

        self.i += 1

    def set_param(self, param):

        self.tau, self.s, self.b, self.m, self.c, self.x = param
