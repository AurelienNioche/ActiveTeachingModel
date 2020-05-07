import numpy as np

EPS = np.finfo(np.float).eps


class Learner:

    bounds = ((0.001, 0.04), (0.2, 0.5))

    def __init__(self, n_item, n_iter_between_ss, n_iter_per_ss, param):

        self.n_item = n_item

        self.c_iter_ss = 0
        self.c_iter = 0
        self.t = 0

        self.n_pres = np.zeros(n_item, dtype=int)
        self.delta = np.zeros(n_item, dtype=int)

        self.seen = np.zeros(n_item, dtype=bool)

        self.n_iter_between_ss = n_iter_between_ss
        self.n_iter_per_ss = n_iter_per_ss

        self.param = np.asarray(param)

        self.heterogeneous_param = len(self.param.shape) > 1

    def update(self, item):

        self.n_pres[item] += 1

        # Increment delta for all items
        self.delta[:] += 1
        # ...except the one for the selected design that equal one
        self.delta[item] = 1

        self.c_iter += 1
        self.c_iter_ss += 1
        self.t += 1
        if self.c_iter_ss >= self.n_iter_per_ss:
            self.delta[:] += self.n_iter_between_ss
            self.t += self.n_iter_between_ss
            self.c_iter_ss = 0

        self.seen[item] = True

    def update_one_step_only(self, item):

        self.delta[:] += 1

        if item is not None:
            self.delta[item] = 1
            self.n_pres[item] += 1
            self.seen[item] = True

    def reply(self, item):

        if self.n_pres[item] == 0:
            return 0

        elif self.delta[item] == 0:
            return 1

        else:
            if self.heterogeneous_param:
                init_forget, rep_effect = self.param[item, :]
            else:
                init_forget, rep_effect = self.param

            fr = init_forget \
                * (1 - rep_effect) ** (self.n_pres[item] - 1)
            p = np.exp(- fr * self.delta[item])

        return np.random.choice([0, 1], p=[1 - p, p])

    def p_seen(self):
        if self.heterogeneous_param:
            init_forget = self.param[self.seen, 0]
            rep_effect = self.param[self.seen, 1]
        else:
            init_forget, rep_effect = self.param

        fr = init_forget \
            * (1-rep_effect) ** (self.n_pres[self.seen] - 1)
        p = np.exp(-fr * self.delta[self.seen])
        return p

    def log_lik(self, item, grid_param):

        assert self.heterogeneous_param is False

        n_param_set = len(grid_param)

        p = np.zeros(n_param_set)

        fr = grid_param[:, 0] \
            * (1 - grid_param[:, 1]) ** (self.n_pres[item] - 1)

        p[:] = np.exp(- fr * self.delta[item])

        p_failure_success = np.zeros((n_param_set, 2))
        p_failure_success[:, 0] = 1 - p
        p_failure_success[:, 1] = p

        log_lik = np.log(p_failure_success + EPS)
        return log_lik

    @classmethod
    def get(cls, tk):

        return cls(
            n_item=tk.n_item,
            n_iter_per_ss=tk.n_iter_per_ss,
            n_iter_between_ss=tk.n_iter_per_ss,
            tk=tk.param)
