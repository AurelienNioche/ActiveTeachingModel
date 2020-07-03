import numpy as np
from . generic import Learner
from scipy.special import expit

EPS = np.finfo(np.float).eps


class ActR2008(Learner):

    def __init__(self, n_item, param):

        self.tau, self.s, self.c, self.a = param

        self.seen = np.zeros(n_item, dtype=bool)
        self.ts = []
        self.hist = []
        self.ds = []

    def p_seen(self, param, is_item_specific, now):

        ts = np.asarray(self.ts)
        hist = np.asarray(self.hist)
        ds = np.asarray(self.ds)

        p = np.zeros(np.sum(self.seen))
        for i, item in enumerate(np.flatnonzero(self.seen)):
            p[i] = self._p(item=item, now=now,
                           ts=ts, hist=hist, ds=ds)
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

    def _p(self, item, now, ts, hist, ds):
        b = hist == item
        rep = ts[b]
        d = ds[b]

        delta = (now - rep)

        e_m = np.sum(np.power(delta, -d))

        x = (-self.tau + np.log(e_m)) / self.s
        p = expit(x)
        return p

    def p(self, item, param, now, is_item_specific):

        hist = np.asarray(self.hist)
        ts = np.asarray(self.ts)
        ds = np.asarray(self.ds)
        b = hist == item
        rep = ts[b]
        d = ds[b]
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

        hist = np.asarray(self.hist)
        ts = np.asarray(self.ts)
        b = hist == item
        rep = ts[b]
        if len(rep) == 0:
            self.ds.append(self.a)
        else:
            ds = np.asarray(self.ds)
            d = ds[b]
            delta = (timestamp - rep)
            e_m = np.sum(np.power(delta, -d))
            d = self.c * e_m + self.a
            self.ds.append(d)

        self.hist.append(item)
        self.ts.append(timestamp)
