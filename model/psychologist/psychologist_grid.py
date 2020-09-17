import numpy as np
from scipy.special import logsumexp

from . generic import Psychologist

EPS = np.finfo(np.float).eps


class PsychologistGrid(Psychologist):

    LIN = 'lin'
    LOG = 'log'

    METHODS = {LIN: np.linspace, LOG: np.geomspace}

    def __init__(self, n_item, is_item_specific, learner,
                 bounds, grid_size, grid_methods, cst_time, true_param=None):

        self.omniscient = true_param is not None
        if not self.omniscient:
            self.bounds = np.asarray(bounds)
            self.methods = np.asarray([self.METHODS[k] for k in grid_methods])
            grid_param = self.cp_grid_param(grid_size=grid_size)

            n_param_set, n_param = grid_param.shape

            lp = np.ones(n_param_set)
            lp -= logsumexp(lp)

            ep = np.dot(np.exp(lp), grid_param)

            if is_item_specific:
                log_post = np.zeros((n_item, n_param_set))
                log_post[:] = lp

                est_param = np.zeros((n_item, n_param))
                est_param[:] = ep

            else:
                log_post = lp
                est_param = ep

            self.grid_param = grid_param
            self.log_post = log_post
            self.est_param = est_param

            self.n_param = n_param

            self.n_pres = np.zeros(n_item, dtype=int)
            self.n_item = n_item

        else:
            self.est_param = true_param

        self.is_item_specific = is_item_specific
        self.cst_time = cst_time
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

        values = np.atleast_2d(
            [m(*b, num=grid_size) for (b, m) in
             zip(self.bounds[diff], self.methods[diff])])

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
                    timestamp=timestamp,
                    cst_time=self.cst_time)

                if self.is_item_specific:
                    lp = self.log_post[item]
                else:
                    lp = self.log_post

                lp += log_lik
                lp -= logsumexp(lp)
                est_param = np.dot(np.exp(lp), self.grid_param)

                if self.is_item_specific:
                    self.log_post[item] = lp
                    self.est_param[item] = est_param
                else:
                    self.log_post = lp
                    self.est_param = est_param

            self.n_pres[item] += 1

        self.learner.update(timestamp=timestamp, item=item)

    def p_seen(self, now, param=None):
        if param is None:
            param = self.est_param

        return self.learner.p_seen(
            param=param,
            is_item_specific=self.is_item_specific,
            cst_time=self.cst_time,
            now=now)

    def inferred_learner_param(self):

        if self.omniscient or not self.is_item_specific:
            return self.est_param

        is_rep = self.n_pres > 1
        not_is_rep = np.invert(is_rep)

        if np.sum(is_rep) == self.n_item or np.sum(not_is_rep) == self.n_item:
            return self.est_param

        # lp_to_consider = self.log_post[is_rep]
        #
        # lp = np.sum(lp_to_consider, axis=0)
        # lp -= logsumexp(lp)
        # est_param = np.dot(np.exp(lp), self.grid_param)
        #
        # self.log_post[not_is_rep] = lp
        self.est_param[not_is_rep] = np.mean(self.est_param[is_rep], axis=0)

        return self.est_param

    def p(self, param, item, now):
        return self.learner.p(
            item=item,
            is_item_specific=self.is_item_specific,
            cst_time=self.cst_time,
            param=param,
            now=now)
