import numpy as np
from scipy.special import expit


class ActR2008:

    def __init__(self, param):

        self.param = param

        self.tau, self.s, self.c, self.a, self.dt = self.param

        self.hist = []
        self.ts = []
        self.ds = []

    def p(self, item, now):

        hist = np.asarray(self.hist)
        ts = np.asarray(self.ts)
        ds = np.asarray(self.ds)
        b = hist == item
        rep = ts[b]
        d = ds[b]
        n = len(rep)
        if n == 0:
            return 0

        delta = (now - rep) * self.dt
        if np.min(delta) == 0:
            return 1

        e_m = np.sum(np.power(delta, -d))

        x = (-self.tau + np.log(e_m)) / self.s
        p = expit(x)
        return p

    def update(self, item, now):
        hist = np.asarray(self.hist)
        ts = np.asarray(self.ts)
        b = hist == item
        rep = ts[b]
        if len(rep) == 0:
            self.ds.append(self.a)
        else:
            ds = np.asarray(self.ds)
            d = ds[b]
            delta = (now - rep) * self.dt
            e_m = np.sum(np.power(delta, -d))
            d = self.c * e_m + self.a
            self.ds.append(d)

        self.hist.append(item)
        self.ts.append(now)
