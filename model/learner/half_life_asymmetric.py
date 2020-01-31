import numpy as np

from . half_life import HalfLife


class HalfLifeAsymmetric(HalfLife):

    bounds = {
        'alpha': (0.00, 1.0),
        'beta': (-1.00, 1.0),
        'gamma': (0.00, 1.0),
    }

    asymmetric = True

    def __init__(self, task_param, param=None):

        super().__init__(task_param=task_param, param=None)

        if param is not None:
            self.alpha, self.beta, self.gamma = \
                param['alpha'], param['beta'],  param['gamma']
        else:
            self.alpha, self.beta, self.gamma = None, None, None

        self.n_success = np.zeros(self.n_item, dtype=int)
        self.n_failure = np.zeros(self.n_item, dtype=int)

        self.i = None
        self.delta_i = None
        self.seen_i = None
        self.n_success_i = None
        self.n_failure_i = None

    def p(self, item):

        assert item is not None, "'item' should be an integer"

        seen = self.seen[item] == 1
        if seen:
            p = np.exp(
                - self.alpha
                * (1 - self.beta) ** self.n_failure[item]
                * (1 - self.gamma) ** self.n_success[item]
                * self.delta[item])
        else:
            p = 0

        return p

    def forgetting_rate_and_p_seen(self):

        fr = \
            self.alpha \
            * (1 - self.beta) ** self.n_failure[self.seen] \
            * (1 - self.gamma) ** self.n_success[self.seen]

        pr = np.exp(-fr*self.delta[self.seen])
        return fr, pr

    def p_grid(self, grid_param, i):

        n_param_set = len(grid_param)
        p = np.zeros((n_param_set, 2))

        i_has_been_seen = self.seen[i] == 1
        if i_has_been_seen:
            p[:, 1] = np.exp(
                - grid_param[:, 0]
                * (1 - grid_param[:, 1]) ** self.n_failure[i]
                * (1 - grid_param[:, 2]) ** self.n_success[i]
                * self.delta[i])

        p[:, 0] = 1 - p[:, 1]
        return p

    def update(self, item, response):

        self.i = item

        if item is not None:
            self.delta_i = self.delta[item]
            self.seen_i = self.seen[item]
            self.n_success_i = self.n_success[item]
            self.n_failure_i = self.n_failure[item]

        # Increment delta for all items
        self.delta += 1

        # ...except the one for the selected design that equal one
        if item is not None:

            if self.seen[item]:
                self.n_success[item] += response
                self.n_failure[item] += (response-1) * -1

            self.delta[item] = 1
            self.seen[item] = 1

    def cancel_update(self):

        if self.i is not None:

            self.n_success[self.i] = self.n_success_i
            self.n_failure[self.i] = self.n_failure_i
            self.seen[self.i] = self.seen_i

        self.delta -= 1

        if self.i is not None:
            self.delta[self.i] = self.delta_i

    def set_param(self, param):
        self.alpha, self.beta, self.gamma = param
