import numpy as np

from learner.generic import Learner


class HalfLife(Learner):

    version = 2.2
    bounds = {
        'alpha': (0.001, 1.0),
        'beta': (0.001, 1.0),
    }

    def __init__(
            self,
            t=0, hist=None,
            n_possible_replies=None,
            param=None,
            known_param=None,
            **kwargs):

        super().__init__(**kwargs)

        self.alpha = None
        self.beta = None

        self.set_cognitive_parameters(param, known_param)

        assert self.alpha is not None
        assert self.beta is not None

        self.b = {}
        self.t_r = {}

        if n_possible_replies:
            self.n_possible_replies = n_possible_replies
            self.p_random = 1 / self.n_possible_replies
        else:
            # raise Exception
            self.p_random = 0

        if hist is not None:
            for t_index in range(t):

                item = hist[t_index]
                self.t_r[item] = t_index

                if item not in self.b:
                    self.b[item] = self.beta
                else:
                    self.b[item] *= (1 - self.alpha)

        self.t = t

        self.old_b = None
        self.old_t_r = None
        self.old_item = None

    def p_recall(self, item, time=None, time_index=None):

        if time_index is not None or time is not None:
            raise NotImplementedError

        if item not in self.t_r:
            return 0

        b = self.b[item]
        t_r = self.t_r[item]

        p = np.exp(-b * (self.t - t_r))
        return p

    def learn(self, item, time=None, time_index=None):

        if time_index is not None or time is not None:
            raise NotImplementedError

        self.old_b = self.b.copy()
        self.old_t_r = self.t_r.copy()

        self.t_r[item] = self.t

        if item not in self.b:
            self.b[item] = self.beta
        else:
            self.b[item] *= (1-self.alpha)

        self.t += 1

    def unlearn(self):

        self.b = self.old_b.copy()
        self.t_r = self.old_t_r.copy()

    def set_history(self, hist, t, times=None):
        raise NotImplementedError

    def _p_choice(self, item, reply, possible_replies, time=None):
        raise NotImplementedError

    def _p_correct(self, item, reply, possible_replies, time=None):
        raise NotImplementedError


class FastHalfLife(Learner):

    bounds = {
        'alpha': (0.001, 1.0),
        'beta': (0.001, 1.0),
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
            raise ValueError

        seen = self.seen[item] == 1
        if seen:
            p = np.exp(
                - self.beta
                * (1 - self.alpha) ** self.n_pres_minus_one[item]
                * self.delta[item])
        else:
            p = 0

        return p

    def p_recalls_and_forgetting_rates(self):

        forgetting_rates = np.full(self.n_item, np.inf)
        p_recalls = np.zeros(self.n_item)

        fr_seen = \
            self.beta * (1 - self.alpha) ** self.n_pres_minus_one[self.seen]

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
