import numpy as np

from . generic import Learner


class HalfLife(Learner):

    bounds = {
        'alpha': (0.00, 1.0),
        'beta': (0.00, 1.0),
    }

    def __init__(self, n_item,
                 t=0, hist=None,
                 param=None,
                 known_param=None,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.alpha = None
        self.beta = None

        self.set_cognitive_parameters(param, known_param)

        self.n_item = n_item

        self.delta = np.zeros(self.n_item, dtype=int)
        self.n_pres_minus_one = np.full(self.n_item, -1, dtype=int)
        self.seen = np.zeros(self.n_item, dtype=bool)

        self.old_delta = None
        self.old_seen = None

        self.alpha = None
        self.beta = None

        self.set_cognitive_parameters(param, known_param)

        assert self.alpha is not None
        assert self.beta is not None

        self.old_b = None
        self.old_t_r = None
        self.old_item = None

        if hist is not None:
            for t_index in range(t):
                item = hist[t_index]
                self.learn(item)

    def p_recall(self, item, time=None, time_index=None):

        if time_index is not None or time is not None:
            raise NotImplementedError

        if item is None:
            raise ValueError("'item' should be an integer")

        seen = self.seen[item] == 1
        if seen:
            p = np.exp(-
                self._forgetting_rate(item)
                * self.delta[item])
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

    def learn(self, item, time=None, time_index=None):

        if time_index is not None or time is not None:
            raise NotImplementedError

        self.old_delta = self.delta[item]
        self.old_seen = self.seen[item]
        self.old_item = item

        self.n_pres_minus_one[item] += 1
        self.delta += 1

        self.delta[item] = 1
        self.seen[item] = 1

    def unlearn(self):

        self.delta -= 1
        self.n_pres_minus_one[self.old_item] -= 1

        self.delta[self.old_item] = self.old_delta
        self.seen[self.old_item] = self.old_seen

    def reset(self):

        self.delta[:] = 0
        self.n_pres_minus_one[:] = -1
        self.seen[:] = 0

    def set_history(self, hist, t, times=None):
        raise NotImplementedError

    def _p_choice(self, item, reply, possible_replies, time=None):
        raise NotImplementedError

    def _p_correct(self, item, reply, possible_replies, time=None):
        raise NotImplementedError

    @classmethod
    def p(cls, grid_param, n_pres_minus_one, seen, delta, i):

        n_param_set = len(grid_param)
        p = np.zeros((n_param_set, 2))

        i_has_been_seen = seen[i] == 1
        if i_has_been_seen:
            p[:, 1] = np.exp(
                - grid_param[:, 0]
                * (1 - grid_param[:, 1]) ** n_pres_minus_one[i]
                * delta[i])

        p[:, 0] = 1 - p[:, 1]
        return p

    @classmethod
    def p_recall_seen(cls, n_pres_minus_one, delta, seen, param):

        alpha, beta = param

        return np.exp(
            - alpha
            * (1 - beta) ** n_pres_minus_one[seen]
            * delta[seen])
