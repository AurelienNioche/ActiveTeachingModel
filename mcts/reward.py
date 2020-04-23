import numpy as np


class Reward:

    def __init__(self, param, n_item):
        self.param = param
        self.n_item = n_item

    def reward(self, n_pres, delta, **kwargs):
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

    def reward(self, n_pres, delta, k=100, x0=0.9, **kwargs):

        p = self.non_zero_p_recall(n_pres=n_pres, delta=delta)
        if len(p):
            return np.sum(1 / (1 + np.exp(-k * (p - x0)))) / self.n_item
        else:
            return 0


class RewardThreshold(Reward):

    def __init__(self, param,  n_item, tau=0.9):
        super().__init__(param=param, n_item=n_item)
        self.tau = tau

    def reward(self, n_pres, delta, normalize=True, **kwargs):

        p = self.non_zero_p_recall(n_pres=n_pres, delta=delta)
        n_learnt = np.sum(p > self.tau)
        if normalize:
            return n_learnt / self.n_item
        else:
            return n_learnt


class RewardAverage(Reward):

    def __init__(self, param,  n_item):
        super().__init__(param=param, n_item=n_item)

    def reward(self, n_pres, delta, **kwargs):

        p = self.non_zero_p_recall(n_pres=n_pres, delta=delta)
        sum_p = np.sum(p)
        return sum_p / self.n_item


class RewardHalfLife(Reward):

    def __init__(self, param,  n_item, max_value=43200 * 10):
        super().__init__(param=param, n_item=n_item)

        self.max_value = max_value
        self.ln_2 = np.log(2)

    # @staticmethod
    # def half_life(p, fr):
    #     max_value = 43200 * 30
    #     if fr > 0:
    #         hl = np.log(2) / fr  # np.log(2 / p) / fr
    #         return min(hl, max_value)
    #     else:
    #         return max_value

    def reward(self, n_pres, delta, **kwargs):

        p, fr = self.non_zero_p_recall(n_pres=n_pres, delta=delta,
                                       return_fr=True)
        if fr is not None:
            hl = np.zeros(len(fr))
            fr_is_zero = fr == 0
            fr_is_non_zero = np.logical_not(fr_is_zero)
            hl[fr_is_zero] = self.max_value
            hl[fr_is_non_zero] = self.ln_2 / fr[fr_is_non_zero]
            return np.sum(hl * p) / self.n_item
        else:
            return 0


class RewardIntegral(Reward):

    def __init__(self, param,  n_item, t_final):
        super().__init__(param=param, n_item=n_item)
        self.t_final = t_final

    def reward(self, n_pres, delta, t=None, **kwargs):
        if t is None:
            raise ValueError('t must be defined')

        # print(self.t_final-t)

        seen = n_pres[:] > 0
        if np.sum(seen) == 0:
            return 0
        n_pres_seen = n_pres[seen]
        delta_seen = delta[seen]
        fr = self.param[0] * (1 - self.param[1]) ** (n_pres_seen - 1)
        delta_to_final = delta_seen + (self.t_final - t)
        integral = - np.exp(-fr * delta_to_final) / delta_to_final + \
            np.exp(-fr*delta_seen) / delta_seen
        return np.sum(integral) / (self.n_item * (self.t_final - t))


class RewardGoal(Reward):

    def __init__(self, param, n_item, tau, t_final):
        super().__init__(param=param, n_item=n_item)
        self.t_final = t_final
        self.tau = tau

    def reward(self, n_pres, delta, t=None, **kwargs):

        seen = n_pres[:] > 0
        if np.sum(seen) == 0:
            return 0
        n_pres_seen = n_pres[seen]
        delta_seen = delta[seen]
        fr = self.param[0] * (1 - self.param[1]) ** (n_pres_seen - 1)

        # fr_is_zero = fr == 0
        # fr_is_non_zero = np.logical_not(fr_is_zero)
        delta_thr = -np.log(self.tau) / fr
        exp_time = self.t_final - ((t - delta_seen) + delta_thr)
        return (np.sum(1/ exp_time[exp_time > 0]) + 1 * np.sum(exp_time < 0))/self.n_item