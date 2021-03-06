import numpy as np


class Myopic:

    def __init__(self, n_item, learnt_threshold):

        self.n_item = n_item
        self.learnt_threshold = learnt_threshold

    def ask(self, psy, now):

        p, seen = psy.p_seen(now)
        min_p = np.min(p)

        if np.sum(seen) == self.n_item or min_p <= self.learnt_threshold:
            item_idx = np.flatnonzero(seen)[np.argmin(p)]
        else:
            item_idx = np.argmin(seen)

        return item_idx
