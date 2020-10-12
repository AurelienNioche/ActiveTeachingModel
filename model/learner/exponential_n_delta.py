import numpy as np

EPS = np.finfo(np.float).eps


class ExponentialNDelta:

    DUMMY_VALUE = -1

    def __init__(self, n_item, n_iter):

        self.n_item = n_item

        self.seen = np.zeros(n_item, dtype=bool)
        self.ts = np.full(n_iter, self.DUMMY_VALUE, dtype=float)
        self.hist = np.full(n_iter, self.DUMMY_VALUE, dtype=int)
        self.seen_item = None
        self.n_seen = 0
        self.i = 0

        self.n_pres = np.zeros(n_item, dtype=int)
        self.last_pres = np.zeros(n_item, dtype=float)

    def p_seen(self, param, is_item_specific, now, cst_time):

        seen = self.n_pres >= 1
        if np.sum(seen) == 0:
            return np.array([]), seen

        if is_item_specific:
            init_forget = param[seen, 0]
            rep_effect = param[seen, 1]
        else:
            init_forget, rep_effect = param

        fr = init_forget * (1 - rep_effect) ** (self.n_pres[seen] - 1)

        last_pres = self.last_pres[seen]
        delta = now - last_pres

        delta *= cst_time
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            p = np.exp(-fr * delta)
        return p, seen

    @staticmethod
    def p_seen_spec_hist(param, now, hist, ts, seen, is_item_specific,
                         cst_time):

        if is_item_specific:
            init_forget = param[seen, 0]
            rep_effect = param[seen, 1]
        else:
            init_forget, rep_effect = param

        seen_item = np.flatnonzero(seen)

        n_seen = np.sum(seen)
        n_pres = np.zeros(n_seen)
        last_pres = np.zeros(n_seen)
        for i, item in enumerate(seen_item):
            is_item = hist == item
            n_pres[i] = np.sum(is_item)
            last_pres[i] = np.max(ts[is_item])

        fr = init_forget * (1-rep_effect) ** (n_pres - 1)

        delta = now - last_pres
        delta *= cst_time
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            p = np.exp(-fr * delta)
        return p, seen

    def log_lik_grid(self, item, grid_param, response, timestamp,
                     cst_time):

        fr = grid_param[:, 0] \
             * (1 - grid_param[:, 1]) ** (self.n_pres[item] - 1)

        delta = timestamp - self.last_pres[item]

        delta *= cst_time
        p_success = np.exp(- fr * delta)

        p = p_success if response else 1-p_success

        log_lik = np.log(p + EPS)
        return log_lik

    def p(self, item, param, now, is_item_specific, cst_time):

        if is_item_specific:
            init_forget = param[item, 0]
            rep_effect = param[item, 1]
        else:
            init_forget, rep_effect = param

        fr = init_forget * (1 - rep_effect) ** (self.n_pres[item] - 1)

        delta = now - self.last_pres[item]

        delta *= cst_time
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            p = np.exp(- fr * delta)
        return p

    def update(self, item, timestamp):

        self.last_pres[item] = timestamp
        self.n_pres[item] += 1

        self.hist[self.i] = item
        self.ts[self.i] = timestamp

        self.seen[item] = True

        self.seen_item = np.flatnonzero(self.seen)
        self.n_seen = np.sum(self.seen)

        self.i += 1
