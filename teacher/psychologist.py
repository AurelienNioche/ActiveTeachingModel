from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.utils import timezone

import numpy as np
from scipy.special import logsumexp
from itertools import product
import pandas as pd

EPS = np.finfo(np.float).eps


class Learner:

    @classmethod
    def p_seen(cls, param, n_pres, is_item_specific, last_pres=None, now=None,
               delta=None):

        seen = n_pres >= 1
        if np.sum(seen) == 0:
            return None, []

        if is_item_specific:
            init_forget = param[seen, 0]
            rep_effect = param[seen, 1]
        else:
            init_forget, rep_effect = param

        fr = init_forget * (1-rep_effect) ** (n_pres[seen] - 1)
        if delta is None:
            last_pres = last_pres[seen]
            delta = now - last_pres
        else:
            delta = delta[seen]
        p = np.exp(-fr * delta)
        return p, seen

    @classmethod
    def log_lik(cls, item, grid_param, response, timestamp,
                n_pres, last_pres):

        fr = grid_param[:, 0] \
            * (1 - grid_param[:, 1]) ** (n_pres[item] - 1)

        delta = timestamp - last_pres[item]
        p_success = np.exp(- fr * delta)

        if response == 1:
            p = p_success
        elif response == 0:
            p = 1 - p_success
        else:
            raise ValueError

        log_lik = np.log(p + EPS)
        return log_lik

    @classmethod
    def p(cls, item, param, n_pres, last_pres, timestamp, is_item_specific):

        if is_item_specific:
            init_forget = param[item, 0]
            rep_effect = param[item, 1]
        else:
            init_forget, rep_effect = param

        fr = init_forget * (1 - rep_effect) ** (n_pres[item] - 1)

        delta = timestamp - last_pres[item]
        return np.exp(- fr * delta)


class Psychologist:
    def __init__(self, n_item, is_item_specific,
                 bounds=None, grid_size=None):
        if bounds is not None:
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

        self.n_item = n_item

        self.n_pres = np.zeros(n_item, dtype=int)
        self.is_item_specific = is_item_specific

        self.last_pres = np.zeros(n_item, dtype=int)

    @staticmethod
    def cp_grid_param(grid_size, bounds):

        return np.asarray(list(
            product(*[
                np.linspace(*b, grid_size)
                for b in bounds])))

    def update(self, item, response, timestamp):

        if self.n_pres[item] == 0:
            pass
        else:
            gp = np.reshape(self.grid_param, (-1, self.n_param))
            log_lik = Learner.log_lik(
                item=item,
                grid_param=gp,
                response=response,
                timestamp=timestamp,
                n_pres=np.asarray(self.n_pres),
                last_pres=pd.Series(self.last_pres)
            )

            # Update prior
            if self.is_item_specific:
                log_post = np.reshape(self.log_post, (self.n_item, -1))
                lp = log_post[item]
                lp += log_lik
                lp -= logsumexp(lp)
                log_post[item] = lp
                self.log_post = list(log_post.flatten())
            else:
                lp = np.asarray(self.log_post)
                lp += log_lik
                lp -= logsumexp(lp)
                self.log_post = list(lp)

        self.update_minimal(item=item, timestamp=timestamp)

    def update_minimal(self, item, timestamp):

        self.last_pres[item] = timestamp
        self.n_pres[item] += 1

    def p_seen(self, now, param=None):
        if param is None:
            param = self.inferred_learner_param()
        return Learner.p_seen(param=param,
                              n_pres=self.n_pres,
                              last_pres=self.last_pres,
                              is_item_specific=self.is_item_specific,
                              now=now)

    def inferred_learner_param(self):

        gp = np.reshape(self.grid_param, (-1, self.n_param))
        if self.is_item_specific:

            param = np.zeros((self.n_item, self.n_param))
            param[:] = self.get_init_guess()
            lp = np.reshape(self.log_post, (self.n_item, -1))
            rep = np.asarray(self.n_pres) > 1
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

    def p(self, param, item, timestamp):
        return Learner.p(
            item=item,
            is_item_specific=self.is_item_specific,
            param=param,
            n_pres=self.n_pres,
            last_pres=self.last_pres,
            timestamp=timestamp)

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
    def create(cls, tk):

        return cls(
            n_item=tk.n_item,
            bounds=None,
            grid_size=None,
            is_item_specific=tk.is_item_specific
        )
