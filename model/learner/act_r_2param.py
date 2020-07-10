import numpy as np
from . generic import Learner
from scipy.special import expit

EPS = np.finfo(np.float).eps


class ActR2param(Learner):

    def __init__(self, n_item, n_iter, *args, **kwargs):

        self.seen = np.zeros(n_item, dtype=bool)
        self.ts = np.full(n_iter, -1, dtype=int)
        self.hist = np.full(n_iter, -1, dtype=int)

        # Only for static param
        # self.ds = np.full(n_iter, -1, dtype=float)

        self.seen_item = None
        self.n_seen = 0
        self.i = 0

    def p(self, item, param, now, is_item_specific):

        c, a = param

        b = self.hist == item
        rep = self.ts[b]
        n = len(rep)
        if n == 0:
            p = 0
        else:
            d = np.zeros(n)
            d[0] = a
            for i in range(1, n):
                delta_rep = rep[i] - rep[:i]
                e_m_rep = np.sum(np.power(delta_rep, -d[:i]))
                d[i] = c * e_m_rep + a  # Using previous em

            delta = now - rep
            with np.errstate(divide="ignore"):
                em = np.sum(np.power(delta, -d))

            x = np.log(em)
            p = expit(x)
        return p

    def p_seen(self, param, is_item_specific, now):

        c, a = param

        e_m = np.zeros(self.n_seen)
        for i_it, item in enumerate(self.seen_item):
            b = self.hist == item
            rep = self.ts[b]
            n = len(rep)
            d = np.zeros(n)
            d[0] = a
            for i_r in range(1, n):
                delta_rep = rep[i_r] - rep[:i_r]
                e_m_rep = np.sum(np.power(delta_rep, -d[:i_r]))
                d[i_r] = c * e_m_rep + a  # Using previous em
            delta = now - rep
            with np.errstate(divide="ignore"):
                e_m[i_it] = np.sum(np.power(delta, -d))

        with np.errstate(divide="ignore"):
            x = np.log(e_m)
        p = expit(x)
        return p, self.seen

    def update(self, item, timestamp):

        self.seen[item] = True
        self.hist[self.i] = item
        self.ts[self.i] = timestamp

        self.n_seen = np.sum(self.seen)
        self.seen_item = np.flatnonzero(self.seen)

        self.i += 1

    def log_lik_grid(self, item, grid_param, response, timestamp):

        b = self.hist == item
        rep = np.hstack((self.ts[b], np.array([timestamp, ])))
        n = len(rep)

        delta = timestamp - rep

        e_m = np.zeros(len(grid_param))

        for i_pr, param in enumerate(grid_param):
            c, a = param
            d = np.zeros(n)
            d[0] = a
            for i_r in range(1, n):
                delta_rep = rep[i_r] - rep[:i_r]
                e_m_rep = np.sum(np.power(delta_rep, -d[:i_r]))
                d[i_r] = c * e_m_rep + a  # Using previous em
            with np.errstate(divide="ignore"):
                e_m[i_pr] = np.sum(np.power(delta, -d))

        with np.errstate(divide="ignore"):
            x = np.log(e_m)
        p = expit(x)

        p = p if response else 1-p
        log_lik = np.log(p + EPS)
        return log_lik

    @staticmethod
    def log_lik(param, hist, success, timestamp):
        c, a = param

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
            x = np.log(e_m)

        p = expit(x)
        failure = np.invert(success)
        p[failure] = 1 - p[failure]
        log_lik = np.log(p + EPS)
        return log_lik.sum()
