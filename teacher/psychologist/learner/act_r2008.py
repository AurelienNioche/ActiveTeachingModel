import numpy as np
from . generic import Learner
from scipy.special import expit

EPS = np.finfo(np.float).eps


class ActR2008(Learner):

    def __init__(self, n_item):

        self.seen = np.zeros(n_item, dtype=bool)
        self.ts = []
        self.hist = []

    def p_seen(self, param, is_item_specific, now):

        param = param[self.seen, :] if is_item_specific else param

        ts = np.asarray(self.ts)
        hist = np.asarray(self.hist)

        p = np.zeros(np.sum(self.seen))
        for i, item in enumerate(np.flatnonzero(self.seen)):
            p[i] = self.p(item=item, param=param, now=now,
                          # We already select a single set of parameter
                          is_item_specific=False,
                          rep=ts[hist == item])
        return p, self.seen

    def log_lik(self, item, grid_param, response, timestamp):

        ts = np.asarray(self.ts)
        hist = np.asarray(self.hist)
        rep = ts[hist == item]

        p = np.zeros(len(grid_param))
        for i, param in enumerate(grid_param):
            p[i] = self.p(
                item=item, param=param, now=timestamp,
                # In this case, we consider one single set of parameter
                is_item_specific=False, rep=rep)

        p = p if response else 1-p
        log_lik = np.log(p + EPS)
        return log_lik

    def p(self, item, param, now, is_item_specific, rep=None):

        if rep is None:
            rep = np.asarray(self.ts)[np.asarray(self.hist) == item]

        tau, s, c, a = param[item, :] if is_item_specific else param

        n = len(rep)
        if n == 0:
            return 0

        delta = now - rep
        if np.min(delta) == 0:
            return 1

        d = np.full(n, a)
        e_m = np.zeros(n)

        for i in range(n):
            if i > 0:
                d[i] += c * e_m[i - 1]
            with np.errstate(over='ignore'):
                e_m[i] = np.sum(np.power(delta[:i + 1], -d[:i + 1]))
        x = (-tau + np.log(e_m[-1])) / s
        p = expit(x)
        return p

    def update(self, item, timestamp):

        self.seen[item] = True
        self.ts.append(timestamp)
        self.hist.append(item)

    @classmethod
    def f(cls, delta, d):
        with np.errstate(over='ignore'):
            return delta ** -d
