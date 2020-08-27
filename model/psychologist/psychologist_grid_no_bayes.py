import numpy as np
from scipy.special import logsumexp

from . generic import Psychologist

EPS = np.finfo(np.float).eps


class PsychologistGridNoBayes(Psychologist):

    def __init__(self, n_item, is_item_specific, learner,
                 bounds, grid_size, true_param=None):

        self.omniscient = true_param is not None
        if not self.omniscient:
            self.bounds = np.asarray(bounds)
            grid_param = self.cp_grid_param(grid_size=grid_size)

            n_param_set, n_param = grid_param.shape

            lp = np.ones(n_param_set)
            lp -= logsumexp(lp)

            ep = np.array([np.mean(b) for b in self.bounds])

            if is_item_specific:

                est_param = np.zeros((n_item, n_param))
                est_param[:] = ep

            else:
                est_param = ep

            self.grid_param = grid_param

            self.est_param = est_param

            self.n_param = n_param

            self.n_pres = np.zeros(n_item, dtype=int)
            self.n_item = n_item

        else:
            self.est_param = true_param

        self.is_item_specific = is_item_specific
        self.learner = learner

    @staticmethod
    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    def cp_grid_param(self, grid_size):
        diff = self.bounds[:, 1] - self.bounds[:, 0] > 0
        not_diff = np.invert(diff)

        values = np.atleast_2d([np.linspace(*b, num=grid_size)
                                for b in self.bounds[diff]])
        var = self.cartesian_product(*values)
        grid = np.zeros((max(1, len(var)), len(self.bounds)))
        if np.sum(diff):
            grid[:, diff] = var
        if np.sum(not_diff):
            grid[:, not_diff] = self.bounds[not_diff, 0]

        return grid

    def update(self, item, response, timestamp):

        if not self.omniscient:
            if self.n_pres[item] == 0:
                pass
            else:
                log_lik = self.learner.log_lik_grid(
                    item=item,
                    grid_param=self.grid_param,
                    response=response,
                    timestamp=timestamp)

                est_param = self.grid_param[np.argmax(log_lik)]

                if self.is_item_specific:
                    self.est_param[item] = est_param
                else:
                    self.est_param = est_param

            self.n_pres[item] += 1

        self.learner.update(timestamp=timestamp, item=item)

    def p_seen(self, now, param=None):

        if param is None:
            param = self.est_param

        return self.learner.p_seen(
            param=param,
            is_item_specific=self.is_item_specific,
            now=now)

    def inferred_learner_param(self):
        return self.est_param

    def p(self, param, item, now):
        return self.learner.p(
            item=item,
            is_item_specific=self.is_item_specific,
            param=param,
            now=now)

    @classmethod
    def create(cls, n_item, bounds,
               is_item_specific, learner, grid_size,
               true_param=None):

        return cls(
            true_param=true_param,
            n_item=n_item,
            bounds=bounds,
            grid_size=grid_size,
            is_item_specific=is_item_specific,
            learner=learner)
