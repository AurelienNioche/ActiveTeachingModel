import numpy as np
from scipy.optimize import minimize

from model.learner.act_r2008 import ActR2008
from . generic import Psychologist

EPS = np.finfo(np.float).eps


class PsychologistGradient(Psychologist):

    def __init__(self, n_item, is_item_specific, learner,
                 omniscient, param, init_guess, bounds):

        self.omniscient = omniscient
        if not omniscient:
            self.inferred_param = init_guess
            self.bounds = bounds
            self.n_pres = np.zeros(n_item, dtype=int)
            self.hist = []
            self.success = []
            self.timestamp = []

        else:
            self.inferred_param = param

        self.is_item_specific = is_item_specific
        self.learner = learner

    def update(self, item, response, timestamp):

        if not self.omniscient:
            self.hist.append(item)
            self.success.append(response)
            self.timestamp.append(timestamp)
            if self.n_pres[item] == 0:
                pass
            else:
                self.inferred_param = self.fit(
                    self.learner.inv_log_lik,
                    init_guess=self.inferred_param,
                    hist=np.asarray(self.hist),
                    success=np.asarray(self.success),
                    timestamp=np.asarray(self.timestamp),
                    bounds=self.bounds
                )

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

    @staticmethod
    def fit(inv_log_lik, init_guess,
            hist, success, timestamp, bounds):
        # relevant = hist != -1
        # hist = hist[relevant]
        # success = success[relevant]
        # timestamp = timestamp[relevant]
        r = minimize(inv_log_lik,
                     x0=init_guess,
                     args=(hist,
                           success,
                           timestamp),
                     bounds=bounds,
                     method='SLSQP')
        return r.x

    def p(self, param, item, now):
        return self.learner.p(
            item=item,
            is_item_specific=self.is_item_specific,
            param=param,
            now=now)

    @classmethod
    def create(cls, tk, omniscient):
        if tk.is_item_specific:
            raise NotImplementedError

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
            init_guess=tk.init_guess,
            omniscient=omniscient,
            n_item=tk.n_item,
            bounds=tk.bounds,
            is_item_specific=tk.is_item_specific,
            param=tk.param,
            learner=learner
        )
