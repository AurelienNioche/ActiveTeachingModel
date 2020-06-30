import numpy as np
from . generic import Learner

EPS = np.finfo(np.float).eps


class ExponentialNDelta(Learner):

    def __init__(self, n_item):

        self.n_pres = np.zeros(n_item, dtype=int)
        self.last_pres = np.zeros(n_item, dtype=int)

    def p_seen(self, param, is_item_specific, now):

        seen = self.n_pres >= 1
        if np.sum(seen) == 0:
            return None, []

        if is_item_specific:
            init_forget = param[seen, 0]
            rep_effect = param[seen, 1]
        else:
            init_forget, rep_effect = param

        fr = init_forget * (1 - rep_effect) ** (self.n_pres[seen] - 1)
        # if delta is None:
        last_pres = self.last_pres[seen]
        delta = now - last_pres
        # else:
        #     delta = delta[seen]
        p = np.exp(-fr * delta)
        return p, seen

    def log_lik(self, item, grid_param, response, timestamp):

        fr = grid_param[:, 0] \
             * (1 - grid_param[:, 1]) ** (self.n_pres[item] - 1)

        delta = timestamp - self.last_pres[item]
        p_success = np.exp(- fr * delta)

        p = p_success if response else 1-p_success

        log_lik = np.log(p + EPS)
        return log_lik

    def p(self, item, param, now, is_item_specific):

        if is_item_specific:
            init_forget = param[item, 0]
            rep_effect = param[item, 1]
        else:
            init_forget, rep_effect = param

        fr = init_forget * (1 - rep_effect) ** (self.n_pres[item] - 1)

        delta = now - self.last_pres[item]
        return np.exp(- fr * delta)

    def update(self, item, timestamp):

        self.last_pres[item] = timestamp
        self.n_pres[item] += 1
