import numpy as np
from . generic import Learner
from scipy.special import expit

EPS = np.finfo(np.float).eps


class ActR2008(Learner):

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

    def p(self, item, param, now, is_item_specific):

        tau, s, c, a = param

        b = self.hist == item
        rep = self.ts[b]
        n = len(rep)
        if n == 0:
            p = 0
        else:
            delta = now - rep
            e_m = self._e_m(delta=delta, rep=rep, n=n, c=c, a=a)

            x = (-tau + np.log(e_m)) / s
            p = expit(x)
        return p

    def p_seen(self, param, is_item_specific, now):

        tau, s, c, a = param

        e_m = np.zeros(self.n_seen)
        for i, item in enumerate(self.seen_item):
            b = self.hist == item
            rep = self.ts[b]
            n = len(rep)
            delta = now - rep
            e_m[i] = self._e_m(delta=delta, rep=rep, n=n, c=c, a=a)

        with np.errstate(divide="ignore"):
            x = (-tau + np.log(e_m)) / s
        p = expit(x)
        return p, self.seen

    @staticmethod
    def _e_m(delta, n, rep, c, a):

        d = np.zeros(n)
        d[0] = a
        for i in range(1, n):
            delta_rep = rep[i] - rep[:i]
            e_m_rep = np.sum(np.power(delta_rep, -d[:i]))
            d[i] = c * e_m_rep + a  # Using previous em

        with np.errstate(divide="ignore"):
            em = np.sum(np.power(delta, -d))
        return em

    def update(self, item, timestamp):

        self.seen[item] = True
        self.hist[self.i] = item
        self.ts[self.i] = timestamp

        self.n_seen = np.sum(self.seen)
        self.seen_item = np.flatnonzero(self.seen)

        self.i += 1

    def log_lik(self, item, grid_param, response, timestamp):

        b = self.hist == item
        rep = np.hstack((self.ts[b], np.array([timestamp, ])))
        n = len(rep)

        delta = timestamp - rep

        e_m = np.zeros(len(grid_param))

        for i_pr, param in enumerate(grid_param):
            _, _, c, a = param
            e_m[i_pr] = self._e_m(delta=delta, n=n, rep=rep, c=c, a=a)

        tau = grid_param[:, 0]
        s = grid_param[:, 1]

        with np.errstate(divide="ignore"):
            x = (-tau + np.log(e_m)) / s
        p = expit(x)

        p = p if response else 1-p
        log_lik = np.log(p + EPS)
        return log_lik

    def p_seen_static_param(self, now):

        e_m = np.zeros(self.n_seen)
        for i, item in enumerate(self.seen_item):
            b = self.hist == item
            rep = self.ts[b]
            d = self.ds[b]
            delta = now - rep
            e_m[i] = np.sum(np.power(delta, -d))

        x = (-self.tau + np.log(e_m)) / self.s
        p = expit(x)
        return p, self.seen

    def update_static_param(self, item, timestamp):

        self.seen[item] = True
        b = self.hist == item
        rep = self.ts[b]
        if len(rep) == 0:
            self.ds[self.i] = self.a
        else:
            d = self.ds[b]
            delta = timestamp - rep
            e_m = np.sum(np.power(delta, -d))
            d = self.c * e_m + self.a
            self.ds[self.i] = d

        self.hist[self.i] = item
        self.ts[self.i] = timestamp

        self.n_seen = np.sum(self.seen)
        self.seen_item = np.flatnonzero(self.seen)

        self.i += 1

    def p_static_param(self, item, now):

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
