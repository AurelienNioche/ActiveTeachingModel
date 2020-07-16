import numpy as np
from . generic import Learner
from scipy.special import expit
import math

EPS = np.finfo(np.float).eps


class Walsh2018(Learner):

    def __init__(self, n_item, n_iter):

        self.seen = np.zeros(n_item, dtype=bool)
        self.ts = np.full(n_iter, -1, dtype=float)
        self.hist = np.full(n_iter, -1, dtype=int)

        self.seen_item = None
        self.n_seen = 0
        self.i = 0

        self.tau = None
        self.s = None
        self.b = None
        self.m = None
        self.c = None
        self.x = None

    def p(self, item, param, now, is_item_specific):

        self.set_param(param=param)

        relevant = self.hist == item
        rep = self.ts[relevant]
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

    def p_seen(self, param, is_item_specific, now):

        self.set_param(param=param)
        _m_ = np.zeros(self.n_seen)
        for i_it, item in enumerate(self.seen_item):

            is_item = self.hist == item
            rep = self.ts[is_item]
            n = len(rep)
            delta = (now - rep)

            if np.min(delta) == 0:
                _m_item = np.inf
            else:
                w = delta ** -self.x
                w /= np.sum(w)

                _t_ = np.sum(w * delta)
                if n > 1:
                    lag = rep[1:] - rep[:-1]
                    d = self.b + self.m * np.mean(1 / np.log(lag + math.e))
                else:
                    d = self.b

                _m_item = n ** self.c * _t_ ** -d

            _m_[i_it] = _m_item

        with np.errstate(divide="ignore", invalid="ignore"):
            v = (-self.tau + _m_) / self.s
            p = expit(v)
        return p, self.seen

    def log_lik_grid(self, item, grid_param, response, timestamp):
        p = np.zeros(len(grid_param))
        for i, param in enumerate(grid_param):
            p[i] = self.p(item=item, param=param, now=timestamp,
                          is_item_specific=False)
        p = p if response else 1 - p
        return np.log(p+EPS)

    @staticmethod
    def log_lik(param, hist, success, timestamp):
        if isinstance(param, dict):
            tau, s, b, m, c, x = \
                param["tau"], param["s"], param["b"], \
                param["m"], param["c"], param["x"]
        else:
            tau, s, b, m, c, x = param

        _m_ = np.zeros(len(hist))

        for item in np.unique(hist):

            is_item = hist == item
            rep = timestamp[is_item]
            n = len(rep)

            _m_item = np.zeros(n)

            _m_item[0] = - np.inf  # To adapt for xp
            if n > 1:
                _m_item[1] = (rep[1]-rep[0])**-b
            for i in range(2, n):
                delta = rep[i] - rep[:i]

                w = delta ** -x
                w /= np.sum(w)

                _t_ = np.sum(w * delta)

                lag = rep[1:i+1] - rep[:i]
                d = b + m * np.mean(1 / np.log(lag + math.e))
                _m_item[i] = i ** c * _t_ ** -d

            _m_[is_item] = _m_item

        with np.errstate(divide="ignore", invalid="ignore"):
            v = (-tau + _m_) / s

        p = expit(v)
        failure = np.invert(success)
        p[failure] = 1 - p[failure]
        # print("hist", hist, "success", success)
        # print("param", param, "p", p)
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

        if isinstance(param, dict):
            for k, v in param.items():
                setattr(self, k, v)
        else:
            self.tau, self.s, self.b, self.m, self.c, self.x = param
