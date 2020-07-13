import numpy as np
from scipy.special import expit
import math


class Walsh2018:

    def __init__(self, tau, s, c, x, b, m, dt):

        self.tau = tau
        self.s = s
        self.c = c
        self.x = x
        self.b = b
        self.m = m
        self.dt = dt

        self.hist = []
        self.ts = []

    def p(self, item, now):

        hist = np.asarray(self.hist)
        ts = np.asarray(self.ts)

        relevant = hist == item
        rep = ts[relevant]
        n = len(rep)
        delta = (now - rep) * self.dt

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
            # print("w", w, "m", _m_, "d", d, "n**c", n**self.c, "t", _t_)

            v = (-self.tau + _m_) / self.s
            p = expit(v)
            return p

    def update(self, item, now):

        self.hist.append(item)
        self.ts.append(now)
