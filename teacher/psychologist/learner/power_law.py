import numpy as np
from . generic import Learner
from scipy.special import expit

EPS = np.finfo(np.float).eps

N_SEC_DAY = 60*60*24


class PowerLaw(Learner):

    def __init__(self, n_item):

        self.seen = np.zeros(n_item, dtype=bool)
        self.timestamps = np.zeros(n_item, dtype=object)
        self.timestamps[:] = [[] for _ in range(n_item)]

    def p_seen(self, param, is_item_specific, now):
        d, tau, temp = param[self.seen, :] if is_item_specific else param

        sum_traces = np.zeros(np.sum(self.seen))
        for i, ts in enumerate(self.timestamps[self.seen]):

            delta = now - np.asarray(ts)
            traces = self.f(delta=delta, d=d)
            sum_traces[i] = traces.sum()

        strength = np.log(sum_traces)
        x = (- tau + strength) / temp
        p = expit(x)
        return p, self.seen

    def log_lik(self, item, grid_param, response, timestamp):

        d = grid_param[:, 0]
        tau = grid_param[:, 1]
        temp = grid_param[:, 2]

        delta = timestamp - np.asarray(self.timestamps[item])
        sum_traces = np.zeros(len(d))
        for i, _d in enumerate(d):
            traces = self.f(delta=delta, d=_d)
            sum_traces[i] = traces.sum(axis=-1)

        strength = np.log(sum_traces)
        x = (- tau + strength) / temp
        p = expit(x)

        p = p if response else 1-p
        log_lik = np.log(p + EPS)
        assert len(log_lik) == len(grid_param)
        return log_lik

    def p(self, item, param, now, is_item_specific):

        d, tau, temp = param[item, :] if is_item_specific else param

        rep = np.asarray(self.timestamps[item])
        if len(rep) == 0:
            return 0

        delta = now - rep
        if np.min(delta) == 0:
            return 1
        traces = self.f(delta=delta, d=d)
        sum_traces = traces.sum()
        strength = np.log(sum_traces)
        x = (- tau + strength) / temp
        p = expit(x)
        return p

    def update(self, item, timestamp):

        self.seen[item] = True
        self.timestamps[item].append(timestamp)

    @classmethod
    def f(cls, delta, d):
        with np.errstate(over='ignore'):
            return (delta * 100) ** -d
