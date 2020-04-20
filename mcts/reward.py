import numpy as np


class Reward:

    def __init__(self, param, n_item):
        self.param = param
        self.n_item = n_item

    def reward(self, n_pres, delta, t):
        pass

    def non_zero_p_recall(self, n_pres, delta):
        seen = n_pres[:] > 0
        if np.sum(seen) == 0:
            return np.array([])

        fr = self.param[0] * (1 - self.param[1]) ** (n_pres[seen] - 1)
        p = np.exp(-fr * delta[seen])
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

    def reward(self, n_pres, delta, t):

        p = self.non_zero_p_recall(n_pres=n_pres, delta=delta)
        n_learnt = np.sum(p > self.tau)
        return n_learnt
        # return n_learnt / self.n_item
