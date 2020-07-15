import numpy as np
from scipy.optimize import minimize

from model.learner.act_r2008 import ActR2008
from model.learner.walsh2018 import Walsh2018
# from model.learner.act_r_2param import ActR2param
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
        self.true_param = param
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

        self.learner.update(timestamp=timestamp, item=item)

    def p_seen(self, now):

        p_seen, seen = self.learner.p_seen(
            param=self.inferred_param,
            is_item_specific=self.is_item_specific,
            now=now)
        # print("hist", self.hist)
        # print("success", self.success)
        # print("p_seen", p_seen)
        return p_seen, seen

    def inferred_learner_param(self):
        return self.inferred_param

    def fit(self, inv_log_lik, init_guess,
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
        param = r.x
        # print("inf param", r.x)
        # print("inf param LLS", - r.fun)
        # print("true param", self.true_param)
        # print("true param LLS", self.learner.log_lik(
        #     param=self.true_param,
        #     hist=hist,
        #     success=success,
        #     timestamp=timestamp))
        # print()
        return param

    def p(self, param, item, now):

        p = self.learner.p(
            item=item,
            is_item_specific=self.is_item_specific,
            param=param,
            now=now)
        return p

    @classmethod
    def create(cls, tk, omniscient):
        if tk.is_item_specific:
            raise NotImplementedError

        if tk.learner_model in (ActR2008, Walsh2018):
            if tk.is_item_specific:
                raise NotImplementedError
            else:
                learner = tk.learner_model(n_item=tk.n_item,
                                           n_iter=tk.n_ss*tk.ss_n_iter
                                           + tk.horizon)
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
