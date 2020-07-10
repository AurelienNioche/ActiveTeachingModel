import numpy as np
from scipy.special import expit

EPS = np.finfo(np.float).eps


class ActR2008:

    def __init__(self, n_item, n_iter, param):

        self.tau, self.s, self.c, self.a = param

        self.seen = np.zeros(n_item, dtype=bool)
        self.ts = np.full(n_iter, -1, dtype=int)
        self.hist = np.full(n_iter, -1, dtype=int)

        # Only for static param
        self.ds = np.full(n_iter, -1, dtype=float)

        self.seen_item = None
        self.n_seen = 0
        self.i = 0

    @staticmethod
    def log_lik(param, hist, success, timestamp):
        tau, s, c, a = param

        e_m = np.zeros(len(hist))

        for item in np.unique(hist):

            b = hist == item
            rep = timestamp[b]
            n = len(rep)

            e_m_item = np.zeros(n)

            d = np.zeros(n - 1)
            e_m_item[0] = 0  # To adapt for xp
            if n > 1:
                d[0] = a
                for i in range(1, n):
                    delta_rep = rep[i] - rep[:i]
                    e_m_item[i] = np.sum(np.power(delta_rep, -d[:i]))
                    if i < n - 1:
                        d[i] = c * e_m[i] + a  # Using previous em

            e_m[b] = e_m_item
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (-tau + np.log(e_m)) / s
        p = expit(x)
        failure = np.invert(success)
        p[failure] = 1 - p[failure]
        log_lik = np.log(p + EPS)
        return log_lik.sum()

    def update(self, item, now):

        self.seen[item] = True
        b = self.hist == item
        rep = self.ts[b]
        if len(rep) == 0:
            self.ds[self.i] = self.a
        else:
            d = self.ds[b]
            delta = now - rep
            e_m = np.sum(np.power(delta, -d))
            d = self.a + self.c * e_m
            self.ds[self.i] = d

        self.hist[self.i] = item
        self.ts[self.i] = now

        self.n_seen = np.sum(self.seen)
        self.seen_item = np.flatnonzero(self.seen)

        self.i += 1

    def p(self, item, now):

        b = self.hist == item
        rep = self.ts[b]
        d = self.ds[b]
        n = len(rep)
        if n == 0:
            return 0

        delta = now - rep
        if np.min(delta) == 0:
            return 1

        e_m = np.sum(np.power(delta, -d))

        x = (-self.tau + np.log(e_m)) / self.s
        p = expit(x)
        return p

    @classmethod
    def inv_log_lik(cls, *args, **kwargs):
        return - cls.log_lik(*args, **kwargs)

    @staticmethod
    def prior(param):
        tau, s, c, a = param
        if s <= 0:
            return 0
        elif c <= 0:
            return 0

        elif a <= 0:
            return 0

        else:
            return 1

