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

        self.n_item = task_param['n_item']

        self.delta = np.zeros(self.n_item, dtype=int)
        self.n_pres_minus_one = np.full(self.n_item, -1, dtype=int)
        self.seen = np.zeros(self.n_item, dtype=bool)

        self.i = None
        self.delta_i = None
        self.seen_i = None

    def p_recall(self, item):

        assert item is not None, "'item' should be an integer"

        seen = self.seen[item] == 1
        if seen:
            p = np.exp(-self._forgetting_rate(item) * self.delta[item])
        else:
            p = 0

        return p

    def _forgetting_rate(self, item):

        return self.alpha \
            * (1 - self.beta) ** self.n_pres_minus_one[item]

    def p_recalls_and_forgetting_rates(self):

        forgetting_rates = np.full(self.n_item, np.inf)
        p_recalls = np.zeros(self.n_item)

        fr_seen = self._forgetting_rate(self.seen)

        forgetting_rates[self.seen] = fr_seen

        p_recalls[self.seen] = np.exp(
            - fr_seen * self.delta[self.seen])
        return p_recalls, forgetting_rates

    def p(self, grid_param, i):

        """
        Probability of recall for a specific item for several parameter sets
        :param grid_param:
        :param i:
        :return:
        """

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

    def p_recall_seen_only(self, param):

        """
        Probability of recall for seen item only
        :param param:
        :return:
        """

        alpha, beta = param

        return np.exp(
            - alpha
            * (1 - beta) ** self.n_pres_minus_one[self.seen]
            * self.delta[self.seen])

    def update(self, item):

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
