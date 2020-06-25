import numpy as np

EPS = np.finfo(np.float).eps


class Learner:

    def __init__(self,
                 n_ss,
                 terminal_t,
                 n_item, n_iter_between_ss, n_iter_per_ss, param,
                 bounds):

        self.n_item = n_item
        self.terminal_t = terminal_t
        self.n_iter_between_ss = n_iter_between_ss
        self.n_iter_per_ss = n_iter_per_ss
        self.n_ss = n_ss

        self.c_iter_ss = 0
        self.c_iter = 0
        self.t = 0

        self.n_pres = np.zeros(n_item, dtype=int)
        self.delta = np.zeros(n_item, dtype=int)

        self.seen = np.zeros(n_item, dtype=bool)

        self.param = np.asarray(param)
        self.bounds = np.asarray(bounds)

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

        p = self.p(item)
        return np.random.choice([0, 1], p=[1 - p, p])

    def p(self, item, param=None):

        if self.n_pres[item] == 0:
            p = 0

        elif self.delta[item] == 0:
            p = 1

        else:
            if param is not None:
                init_forget, rep_effect = param
            else:
                if self.heterogeneous_param:
                    init_forget, rep_effect = self.param[item, :]
                else:
                    init_forget, rep_effect = self.param

            fr = init_forget \
                * (1 - rep_effect) ** (self.n_pres[item] - 1)
            p = np.exp(- fr * self.delta[item])
        return p

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

    def log_lik(self, item, grid_param, response):

        fr = grid_param[:, 0] \
            * (1 - grid_param[:, 1]) ** (self.n_pres[item] - 1)

        p_success = np.exp(- fr * self.delta[item])

        if response == 1:
            p = p_success
        elif response == 0:
            p = 1 - p_success
        else:
            raise ValueError

        log_lik = np.log(p + EPS)
        return log_lik

    @classmethod
    def get(cls, tk):

        return cls(
            bounds=tk.bounds,
            param=tk.param,
            n_item=tk.n_item,
            n_ss=tk.n_ss,
            n_iter_per_ss=tk.n_iter_per_ss,
            n_iter_between_ss=tk.n_iter_between_ss,
            terminal_t=tk.terminal_t
        )
