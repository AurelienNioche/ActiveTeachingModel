import numpy as np
from scipy.special import logsumexp
from itertools import product

from model.learner.act_r2008 import ActR2008
from . generic import Psychologist

EPS = np.finfo(np.float).eps


class PsychologistGrid(Psychologist):

    def __init__(self, n_item, is_item_specific, learner,
                 bounds, grid_size, omniscient, param):

        self.omniscient = omniscient
        if not omniscient:
            grid_param = self.cp_grid_param(grid_size=grid_size,
                                            bounds=bounds)
            n_param_set = grid_param.shape[0]
            grid_param = grid_param.flatten()

            lp = np.ones(n_param_set)
            lp -= logsumexp(lp)
            if is_item_specific:
                log_post = np.zeros((n_item, n_param_set))
                log_post[:] = lp
                log_post = log_post.flatten()
            else:
                log_post = lp

            self.grid_param = grid_param
            self.log_post = log_post

            self.n_param = len(bounds)
            self.bounds = np.asarray(bounds).flatten()
            self.inferred_param = self.get_init_guess()

            self.n_pres = np.zeros(n_item, dtype=int)
            self.n_item = n_item

        else:
            self.inferred_param = param

        self.is_item_specific = is_item_specific
        self.learner = learner

    @staticmethod
    def cp_grid_param(grid_size, bounds):

        return np.asarray(list(
            product(*[
                np.linspace(*b, grid_size)
                for b in bounds])))

    def update(self, item, response, timestamp):

        if not self.omniscient:
            if self.n_pres[item] == 0:
                pass
            else:
                from datetime import datetime
                gp = np.reshape(self.grid_param, (-1, self.n_param))

                t = datetime.now()
                log_lik = self.learner.log_lik_grid(
                    item=item,
                    grid_param=gp,
                    response=response,
                    timestamp=timestamp)
                print("log_lik", datetime.now()-t)
                # Update prior
                if self.is_item_specific:
                    log_post = np.reshape(self.log_post, (self.n_item, -1))
                    lp = log_post[item]
                    lp += log_lik
                    lp -= logsumexp(lp)
                    log_post[item] = lp
                    self.log_post = log_post.flatten()
                else:
                    lp = np.asarray(self.log_post)
                    lp += log_lik
                    lp -= logsumexp(lp)
                    self.log_post = lp

            self.n_pres[item] += 1
        self.update_learner(item=item, timestamp=timestamp)

    def update_learner(self, item, timestamp):

        self.learner.update(timestamp=timestamp, item=item)

    def p_seen(self, now):

        param = self.inferred_learner_param()
        return self.learner.p_seen(
            param=param,
            is_item_specific=self.is_item_specific,
            now=now)

    def inferred_learner_param(self):

        if not self.omniscient:

            gp = np.reshape(self.grid_param, (-1, self.n_param))
            if self.is_item_specific:

                param = np.zeros((self.n_item, self.n_param))
                param[:] = self.get_init_guess()
                lp = np.reshape(self.log_post, (self.n_item, -1))
                rep = self.n_pres > 1
                param[rep] = gp[lp[rep].argmax(axis=-1)]
            else:
                if np.max(self.n_pres) <= 1:
                    self.inferred_param = self.get_init_guess()
                else:
                    self.inferred_param = gp[np.argmax(self.log_post)]

        return self.inferred_param

    def get_init_guess(self):
        bounds = np.reshape(self.bounds, (-1, 2))
        return [np.mean(b) for b in bounds]

    def p(self, param, item, now):
        return self.learner.p(
            item=item,
            is_item_specific=self.is_item_specific,
            param=param,
            now=now)

    @classmethod
    def generate_param(cls, param, bounds, n_item):

        if isinstance(param, str):
            if param in ("heterogeneous", "het"):
                param = np.zeros((n_item, len(bounds)))
                for i, b in enumerate(bounds):
                    param[:, i] = np.random.uniform(b[0], b[1], size=n_item)

            else:
                raise ValueError
        else:
            param = np.asarray(param)
        return param

    @classmethod
    def create(cls, tk, omniscient):
        if tk.learner_model == ActR2008:
            if tk.is_item_specific:
                raise NotImplementedError
            else:
                learner = tk.learner_model(n_item=tk.n_item,
                                           n_iter=tk.n_ss*tk.ss_n_iter
                                           + tk.horizon,
                                           param=tk.param)
        else:
            learner = tk.learner_model(tk.n_item)
        # else:
        #     learner = tk.learner_model(tk.n_ss * tk.ss_n_iter)
        return cls(
            omniscient=omniscient,
            n_item=tk.n_item,
            bounds=tk.bounds,
            grid_size=tk.grid_size,
            is_item_specific=tk.is_item_specific,
            param=tk.param,
            learner=learner
        )
