import numpy as np

from . generic import Learner


class HalfLife(Learner):

    bounds = {
        'alpha': (0.00, 1.0),
        'beta': (0.00, 1.0),
    }

    def __init__(self, task_param, param=None):

        super().__init__()

        if param is not None:
            self.alpha, self.beta = param['alpha'], param['beta']
        else:
            self.alpha, self.beta = None, None

        self.n_item = task_param['n_item']

        self.delta = np.zeros(self.n_item, dtype=int)
        self.n_pres_minus_one = np.full(self.n_item, -1, dtype=int)
        self.seen = np.zeros(self.n_item, dtype=bool)

        self.i = None
        self.delta_i = None
        self.seen_i = None

    def p(self, item):

        assert item is not None, "'item' should be an integer"

        seen = self.seen[item] == 1
        if seen:
            p = np.exp(
                -self.alpha
                * (1 - self.beta) ** self.n_pres_minus_one[item]
                * self.delta[item])
        else:
            p = 0

        return p

    def forgetting_rate_and_p_seen(self):

        fr = self.alpha \
            * (1 - self.beta) ** self.n_pres_minus_one[self.seen]

        pr = np.exp(-fr*self.delta[self.seen])
        return fr, pr

    def forgetting_rate_and_p_all(self):

        forgetting_rates = np.full(self.n_item, np.inf)
        p_recalls = np.zeros(self.n_item)

        fr_seen, pr_seen = self.forgetting_rate_and_p_seen()

        forgetting_rates[self.seen] = fr_seen
        p_recalls[self.seen] = pr_seen

        return p_recalls, forgetting_rates

    def p_grid(self, grid_param, i):

        n_param_set = len(grid_param)
        p = np.zeros((n_param_set, 2))

        i_has_been_seen = self.seen[i] == 1
        if i_has_been_seen:
            p[:, 1] = np.exp(
                - grid_param[:, 0]
                * (1 - grid_param[:, 1]) ** self.n_pres_minus_one[i]
                * self.delta[i])

        p[:, 0] = 1 - p[:, 1]
        return p

    def update(self, item, response):

        self.i = item
        self.delta_i = self.delta[item]
        self.seen_i = self.seen[item]

        # Increment delta for all items
        self.delta += 1

        # ...except the one for the selected design that equal one
        self.delta[item] = 1
        self.seen[item] = 1
        self.n_pres_minus_one[item] += 1

    def cancel_update(self):

        self.n_pres_minus_one[self.i] -= 1
        self.delta -= 1
        self.delta[self.i] = self.delta_i
        self.seen[self.i] = self.seen_i

    def set_param(self, param):
        self.alpha, self.beta = param

