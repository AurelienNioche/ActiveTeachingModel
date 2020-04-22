import numpy as np


class Reward:

    def __init__(self, param, n_item):
        self.param = param
        self.n_item = n_item

    def reward(self, n_pres, delta, t):
        pass

    def non_zero_p_recall(self, n_pres, delta, return_fr=False):
        seen = n_pres[:] > 0
        if np.sum(seen) == 0:
            p = np.array([])
            fr = None
        else:
            fr = self.param[0] * (1 - self.param[1]) ** (n_pres[seen] - 1)
            p = np.exp(-fr * delta[seen])
        if return_fr:
            return p, fr
        else:
            return p


class RewardSigmoid(Reward):

    def __init__(self, param, n_item, k=100, x0=0.9):

        super().__init__(param=param, n_item=n_item)
        self.k, self.x0 = k, x0

    def reward(self, n_pres, delta, t, k=100, x0=0.9):

        p = self.non_zero_p_recall(n_pres=n_pres, delta=delta)
        if len(p):
            return np.sum(1 / (1 + np.exp(-k * (p - x0)))) / self.n_item
        else:
            return 0


class RewardThreshold(Reward):

    def __init__(self, param,  n_item, tau=0.9):
        super().__init__(param=param, n_item=n_item)
        self.tau = tau

    def reward(self, n_pres, delta, t, normalize=True):

        p = self.non_zero_p_recall(n_pres=n_pres, delta=delta)
        n_learnt = np.sum(p > self.tau)
        if normalize:
            return n_learnt / self.n_item
        else:
            return n_learnt


class RewardHalfLife(Reward):

    def __init__(self, param,  n_item):
        super().__init__(param=param, n_item=n_item)

    @staticmethod
    def half_life(p, fr):
        max_value = 43200 * 30
        if fr > 0:
            hl = np.log(2 / p) / fr
            return min(hl, max_value)
        else:
            return max_value

    def reward(self, n_pres, delta, t):

        p, fr = self.non_zero_p_recall(n_pres=n_pres, delta=delta,
                                       return_fr=True)
        if len(p) and fr is not None:
            hl = [self.half_life(p_i, fr_i) for p_i, fr_i in zip(p, fr)]
            return np.sum(hl*p)
        else:
            return 0
