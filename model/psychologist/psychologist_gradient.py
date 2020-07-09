import numpy as np
from scipy.special import logsumexp
from itertools import product

from model.learner.act_r2008 import ActR2008
from . generic import Psychologist

EPS = np.finfo(np.float).eps


class PsychologistGradient(Psychologist):

    def __init__(self, n_item, is_item_specific, learner,
                 omniscient, param, init_guess):

        self.omniscient = omniscient
        if not omniscient:
            self.inferred_param = init_guess
            self.n_pres = np.zeros(n_item, dtype=int)
            self.hist = []
            self.r = []

        else:
            self.inferred_param = param

        self.init_guess = init_guess
        self.is_item_specific = is_item_specific
        self.learner = learner

    def update(self, item, response, timestamp):

        if not self.omniscient:
            self.hist.append(item)
            self.r.append(response)
            if self.n_pres[item] == 0:
                pass
            else:
                pass

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
        return self.inferred_param

    def p(self, param, item, now):
        return self.learner.p(
            item=item,
            is_item_specific=self.is_item_specific,
            param=param,
            now=now)

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

        return cls(
            omniscient=omniscient,
            n_item=tk.n_item,
            bounds=tk.bounds,
            grid_size=tk.grid_size,
            is_item_specific=tk.is_item_specific,
            param=tk.param,
            learner=learner
        )
