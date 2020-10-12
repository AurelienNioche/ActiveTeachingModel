import numpy as np
from .generic import Learner
from scipy.special import expit
import math

EPS = np.finfo(np.float).eps


class Walsh2018(Learner):
    def __init__(self, n_item, n_iter, cst_time):

        self.seen = np.zeros(n_item, dtype=bool)
        self.ts = np.full(n_iter, -1, dtype=float)
        self.hist = np.full(n_iter, -1, dtype=int)

        self.seen_item = None
        self.n_seen = 0
        self.i = 0

        self.tau = None
        self.s = None
        self.b = None
        self.m = None
        self.c = None
        self.x = None

        self.cst_time = cst_time

    def p(self, item, param, now, is_item_specific, cst_time):

        if len(param.shape) > 1:
            tau, s, b, m, c, x = param[item]

        else:
            tau, s, b, m, c, x = param

        relevant = self.hist == item
        rep = self.ts[relevant]

        rep *= self.cst_time
        now *= self.cst_time

        n = len(rep)
        delta = now - rep

        if n == 0:
            return 0
        elif np.min(delta) == 0:
            return 1
        else:

            w = delta ** -x
            w /= np.sum(w)

            _t_ = np.sum(w * delta)
            if n > 1:
                lag = rep[1:] - rep[:-1]
                d = b + m * np.mean(1 / np.log(lag + math.e))
            else:
                d = b

            _m_ = n ** c * _t_ ** -d

            v = (-tau + _m_) / s
            p = expit(v)
            return p

    def p_seen(self, param, is_item_specific, now, cst_time):

        return self.p_seen_spec_hist(
            param=param,
            now=now,
            hist=self.hist,
            ts=self.ts,
            seen=self.seen,
            is_item_specific=is_item_specific,
            cst_time=cst_time)

    @staticmethod
    def p_seen_spec_hist(param, now, hist, ts, seen, is_item_specific,
                         cst_time):

        if is_item_specific:
            tau = param[seen, 0]
            s = param[seen, 1]
            b = param[seen, 2]
            m = param[seen, 3]
            c = param[seen, 4]
            x = param[seen, 5]

        else:
            tau, s, b, m, c, x = param

        _ts = ts.copy()
        _ts *= cst_time
        now *= cst_time

        n_seen = np.sum(seen)
        n = np.zeros(n_seen)
        _t_ = np.zeros(n_seen)
        mean_lag = np.zeros(n_seen)

        for i_it, item in enumerate(np.flatnonzero(seen)):

            if is_item_specific:
                _x = x[i_it]
            else:
                _x = x

            is_item = hist == item
            rep = _ts[is_item]

            n_it = len(rep)

            delta = now - rep

            w = delta ** -_x
            w /= np.sum(w)

            _t_it = np.sum(w * delta)

            _t_[i_it] = _t_it
            n[i_it] = n_it

            if n_it > 1:
                lag = rep[1:] - rep[:-1]
                mean_lag[i_it] = np.mean(1 / np.log(lag + math.e))

        one_view = n == 1
        more_than_one = np.invert(one_view)

        if is_item_specific:
            b_one_view = b[one_view]
            b_more_than_one = b[more_than_one]
            c = c[more_than_one]
            m = m[more_than_one]
        else:
            b_one_view = b_more_than_one = b

        _m_ = np.zeros(n_seen)
        _m_[one_view] = _t_[one_view] ** -b_one_view
        _m_[more_than_one] = n[more_than_one] ** c * _t_[more_than_one] ** -(
            b_more_than_one + m * mean_lag[more_than_one])

        with np.errstate(divide="ignore", invalid="ignore"):
            v = (-tau + _m_) / s
            p = expit(v)

        return p, seen

    def log_lik_grid(self, item, grid_param, response, timestamp,
                     cst_time):
        p = np.zeros(len(grid_param))
        for i, param in enumerate(grid_param):
            p[i] = self.p(item=item, param=param, now=timestamp,
                          is_item_specific=False, cst_time=cst_time)
        p = p if response else 1 - p
        return np.log(p + EPS)

    def update(self, item, timestamp):

        self.seen[item] = True
        self.hist[self.i] = item
        self.ts[self.i] = timestamp

        self.n_seen = np.sum(self.seen)
        self.seen_item = np.flatnonzero(self.seen)

        self.i += 1
    #
    # def log_lik(self, param, hist, success, timestamp):
    #
    #     tau, s, b, m, c, x = param
    #
    #     _ts = timestamp.copy()
    #     _ts *= self.cst_time
    #
    #     _m_ = np.zeros(len(hist))
    #
    #     for item in np.unique(hist):
    #
    #         is_item = hist == item
    #         rep = _ts[is_item]
    #         n = len(rep)
    #
    #         _m_item = np.zeros(n)
    #
    #         _m_item[0] = -np.inf  # To adapt for xp
    #         if n > 1:
    #             _m_item[1] = (rep[1] - rep[0]) ** -b
    #         for i in range(2, n):
    #             delta = rep[i] - rep[:i]
    #
    #             w = delta ** -x
    #             w /= np.sum(w)
    #
    #             _t_ = np.sum(w * delta)
    #
    #             lag = rep[1: i + 1] - rep[:i]
    #             d = b + m * np.mean(1 / np.log(lag + math.e))
    #             _m_item[i] = i ** c * _t_ ** -d
    #
    #         _m_[is_item] = _m_item
    #
    #     with np.errstate(divide="ignore", invalid="ignore"):
    #         v = (-tau + _m_) / s
    #
    #     p = expit(v)
    #     failure = np.invert(success)
    #     p[failure] = 1 - p[failure]
    #
    #     log_lik = np.log(p + EPS)
    #     return log_lik.sum()
