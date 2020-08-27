import numpy as np
from scipy.optimize import minimize

from model.learner.act_r2008 import ActR2008
from model.learner.walsh2018 import Walsh2018
# from model.learner.act_r_2param import ActR2param
from . generic import Psychologist

EPS = np.finfo(np.float).eps


class PsychologistGradient(Psychologist):

    def __init__(self, n_item, is_item_specific, learner,
                 bounds, true_param=None):

        self.omniscient = true_param is not None
        if not self.omniscient:
            self.bounds = bounds

            ep = np.array([np.mean(b) for b in self.bounds])
            if is_item_specific:
                n_param = len(ep)
                self.inferred_param = np.zeros((n_item, n_param))
                self.est_param[:] = ep
            else:
                self.est_param = ep

            self.n_pres = np.zeros(n_item, dtype=int)
            self.hist = []
            self.success = []
            self.timestamp = []

        else:
            self.est_param = true_param

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
                hist = np.asarray(self.hist)
                ts = np.asarray(self.timestamp)
                success = np.asarray(self.success)

                if self.is_item_specific:
                    is_item = hist == item
                    ep = self.fit(
                        self.learner.inv_log_lik,
                        init_guess=self.inferred_param[item],
                        hist=hist[is_item],
                        success=success[is_item],
                        timestamp=ts[is_item],
                        bounds=self.bounds)

                    self.est_param[item] = ep
                else:
                    ep = self.fit(
                        self.learner.inv_log_lik,
                        init_guess=self.inferred_param,
                        hist=np.asarray(self.hist),
                        success=np.asarray(self.success),
                        timestamp=np.asarray(self.timestamp),
                        bounds=self.bounds)

                    self.est_param = ep

            self.n_pres[item] += 1

        self.learner.update(timestamp=timestamp, item=item)

    def p_seen(self, now, param=None):

        if param is None:
            param = self.est_param

        p_seen, seen = self.learner.p_seen(
            param=param,
            is_item_specific=self.is_item_specific,
            now=now)

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
    def create(cls, learner, n_item, bounds,
               is_item_specific,
               true_param=None):

        return cls(
            true_param=true_param,
            n_item=n_item,
            bounds=bounds,
            is_item_specific=is_item_specific,
            learner=learner)
