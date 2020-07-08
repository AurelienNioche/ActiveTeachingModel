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
    def log_lik(data, param):
        hist, success, timestamp, = data
        tau, s, c, a = param

        p_hist = np.zeros(len(hist))

        for item in np.unique(hist):

            b = hist == item
            rep = timestamp[b]
            n = len(rep)

            p_item = np.zeros(n)

            d = np.zeros(n - 1)
            d[0] = a
            p_item[0] = 0  # To adapt for xp
            for i in range(1, n):
                delta_rep = rep[i] - rep[:i]
                e_m_rep = np.sum(np.power(delta_rep, -d[:i]))
                x = (-tau + np.log(e_m_rep)) / s
                p_item[i] = expit(x)
                if i < n - 1:
                    d[i] = c * e_m_rep + a  # Using previous em

            p_hist[b] = p_item

        failure = np.invert(success)
        p_hist[failure] = 1 - p_hist[failure]
        log_lik = np.log(p_hist + EPS)
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
            d = self.c * e_m + self.a
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

    @staticmethod
    def prior(param):
        tau, s, c, a = param
        if s <= 0:
            return 0
        elif c <= 0:
            return 0

        else:
            return 1
