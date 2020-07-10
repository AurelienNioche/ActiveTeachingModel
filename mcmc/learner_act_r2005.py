import numpy as np
from scipy.special import expit

EPS = np.finfo(np.float).eps


class ActR2005:

    def __init__(self, n_item, n_iter, param):

        self.tau, self.s, self.d = param

        self.seen = np.zeros(n_item, dtype=bool)
        self.ts = np.full(n_iter, -1, dtype=int)
        self.hist = np.full(n_iter, -1, dtype=int)

        self.seen_item = None
        self.n_seen = 0
        self.i = 0

    @staticmethod
    def log_lik(param, hist, success, timestamp):
        tau, s, d = param

        e_m = np.zeros(len(hist))

        for item in np.unique(hist):

            b = hist == item
            rep = timestamp[b]
            n = len(rep)

            e_m_item = np.zeros(n)

            e_m_item[0] = 0  # To adapt for xp
            if n > 1:
                for i in range(1, n):
                    delta_rep = rep[i] - rep[:i]
                    e_m_item[i] = np.sum(np.power(delta_rep, -d))

            e_m[b] = e_m_item

        with np.errstate(divide="ignore", invalid="ignore"):
            x = (-tau + np.log(e_m)) / s
        p = expit(x)
        failure = np.invert(success)
        p[failure] = 1 - p[failure]
        log_lik = np.log(p + EPS)
        return log_lik.sum()

    def p(self, item, now):

        b = self.hist == item
        rep = self.ts[b]
        n = len(rep)
        if n == 0:
            return 0

        delta = now - rep
        if np.min(delta) == 0:
            return 1

        e_m = np.sum(np.power(delta, -self.d))

        x = (-self.tau + np.log(e_m)) / self.s
        p = expit(x)
        return p

    def update(self, item, now):

        self.seen[item] = True

        self.hist[self.i] = item
        self.ts[self.i] = now

        self.n_seen = np.sum(self.seen)
        self.seen_item = np.flatnonzero(self.seen)

        self.i += 1

    @classmethod
    def inv_log_lik(cls, *args, **kwargs):
        return - cls.log_lik(*args, **kwargs)

    @staticmethod
    def prior(param):
        tau, s, d = param
        if 0 <= tau <= 1 and 0 <= s <= 1 and 0 <= d <= 1:
            return 1
        else:
            return 0

