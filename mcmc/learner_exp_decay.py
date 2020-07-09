import numpy as np

EPS = np.finfo(np.float).eps


class ExpDecay:

    def __init__(self, n_iter, param):

        self.param = param

        self.ts = np.full(n_iter, -1, dtype=int)
        self.hist = np.full(n_iter, -1, dtype=int)

        self.n_seen = 0
        self.i = 0

    @classmethod
    def log_lik(cls, param, hist, success, timestamp):
        a, b = param

        log_p_hist = np.zeros(len(hist))

        for item in np.unique(hist):

            is_item = hist == item
            rep = timestamp[is_item]
            n = len(rep)

            log_p_item = np.zeros(n)
            log_p_item[0] = -np.inf  # whatever the model, p=0 // To adapt for xp
            for i in range(1, n):
                delta_rep = rep[i] - rep[i-1]
                fr = a * (1 - b) ** (i - 1)
                log_p_item[i] = -fr * delta_rep

            log_p_hist[is_item] = log_p_item
        p_hist = np.exp(log_p_hist)
        failure = np.invert(success)
        p_hist[failure] = 1 - p_hist[failure]
        log_lik = np.log(p_hist + EPS)
        sum_ll = log_lik.sum()
        return sum_ll

    @classmethod
    def inv_log_lik(cls, *args, **kwargs):
        return - cls.log_lik(*args, **kwargs)

    def update(self, item, now):

        self.hist[self.i] = item
        self.ts[self.i] = now

        self.i += 1

    def p(self, item, now):

        b = self.hist == item
        rep = self.ts[b]
        n = len(rep)
        if n == 0:
            return 0

        delta = now - rep[-1]
        if delta == 0:
            return 1
        a, b = self.param
        fr = a * (1 - b) ** (n - 1)
        log_p = -fr * delta
        p = np.exp(log_p)
        return p

    @staticmethod
    def prior(param):
        a, b = param
        if a > 0 and 0 <= b <= 1:
            return 1
        else:
            return 0
