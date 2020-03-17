import numpy as np


class ThresholdTeacher:

    def __init__(self, n_item, learnt_threshold):

        self.n_pres = np.zeros(n_item, dtype=int)
        self.delta = np.zeros(n_item, dtype=int)

        self.learnt_threshold = learnt_threshold
        self.t = 0

    def ask(self, param):

        n_item = len(self.n_pres)
        seen = self.n_pres[:] > 0
        n_seen = np.sum(seen)

        items = np.arange(n_item)

        if n_seen == 0:
            item = 0

        else:
            fr = param[0] * (1 - param[1]) ** (self.n_pres[seen] - 1)
            p = np.exp(-fr * self.delta[seen])

            min_p = np.min(p)

            if n_seen == n_item or min_p <= self.learnt_threshold:
                item = np.random.choice(items[seen][p[:] == min_p])

            else:
                unseen = np.logical_not(seen)
                item = items[unseen][0]

        self.n_pres[item] += 1

        # Increment delta for all items
        self.delta[:] += 1
        # ...except the one for the selected design that equal one
        self.delta[item] = 1

        return item
