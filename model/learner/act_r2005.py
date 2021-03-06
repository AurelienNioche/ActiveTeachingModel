import numpy as np
from scipy.special import expit

EPS = np.finfo(np.float).eps


class ActR2005:

    def __init__(self, n_item):

        self.seen = np.zeros(n_item, dtype=bool)
        self.timestamps = np.zeros(n_item, dtype=object)
        self.timestamps[:] = [[] for _ in range(n_item)]

    def p_seen(self, param, is_item_specific, now):

        param = param[self.seen, :] if is_item_specific else param

        p = np.zeros(np.sum(self.seen))
        for i, item in enumerate(np.flatnonzero(self.seen)):
            p[i] = self.p(item=item, param=param, now=now,
                          # We already select a single set of parameter
                          is_item_specific=False)
        return p, self.seen

    def log_lik_grid(self, item, grid_param, response, timestamp):

        p = np.zeros(len(grid_param))
        for i, param in enumerate(grid_param):
            p[i] = self.p(
                item=item, param=param, now=timestamp,
                # In this case, we consider one single set of parameter
                is_item_specific=False)

        p = p if response else 1-p
        log_lik = np.log(p + EPS)
        return log_lik

    def p(self, item, param, now, is_item_specific):

        tau, s, a = param[item, :] if is_item_specific else param

        rep = np.asarray(self.timestamps[item])
        n = len(rep)
        if n == 0:
            return 0

        delta = now - rep
        if np.min(delta) == 0:
            return 1

        with np.errstate(over='ignore'):
            m = np.log(np.sum(np.power(delta, -a)))
        x = (-tau + m) / s
        p = expit(x)
        return p

    def update(self, item, timestamp):

        self.seen[item] = True
        self.timestamps[item].append(timestamp)

    @classmethod
    def f(cls, delta, d):
        with np.errstate(over='ignore'):
            return delta ** -d
