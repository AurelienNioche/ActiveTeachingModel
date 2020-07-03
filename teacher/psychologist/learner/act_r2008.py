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
        self.ds = np.full(n_iter, -1, dtype=float)

        self.i = 0

    def p_seen(self, param, is_item_specific, now):

        e_m = np.zeros(np.sum(self.seen))
        for i, item in enumerate(np.flatnonzero(self.seen)):
            e_m[i] = self._e_m(item=item, now=now)

        x = (-self.tau + np.log(e_m)) / self.s
        p = expit(x)
        return p, self.seen

    def log_lik(self, item, grid_param, response, timestamp):
        raise NotImplementedError
        # ts = np.asarray(self.ts)
        # hist = np.asarray(self.hist)
        # ds = np.asarray(self.ds)
        #
        # p = np.zeros(len(grid_param))
        # for i, param in enumerate(grid_param):
        #     p[i] = self._p(
        #         item=item, param=param, now=timestamp,
        #         ts=ts, hist=hist, ds=ds)
        #
        # p = p if response else 1-p
        # log_lik = np.log(p + EPS)
        # return log_lik

    def _e_m(self, item, now):
        b = self.hist == item
        rep = self.ts[b]
        d = self.ds[b]

        delta = (now - rep)

        e_m = np.sum(np.power(delta, -d))
        return e_m

    def p(self, item, param, now, is_item_specific):

        b = self.hist == item
        rep = self.ts[b]
        d = self.ds[b]
        n = len(rep)
        if n == 0:
            return 0

        delta = (now - rep)
        if np.min(delta) == 0:
            return 1

        e_m = np.sum(np.power(delta, -d))

        x = (-self.tau + np.log(e_m)) / self.s
        p = expit(x)
        return p

    def update(self, item, timestamp):

        self.seen[item] = True
        b = self.hist == item
        rep = self.ts[b]
        if len(rep) == 0:
            self.ds[self.i] = self.a
        else:
            d = self.ds[b]
            delta = (timestamp - rep)
            e_m = np.sum(np.power(delta, -d))
            d = self.c * e_m + self.a
            self.ds[self.i] = d

        self.hist[self.i] = item
        self.ts[self.i] = timestamp

        self.i += 1
