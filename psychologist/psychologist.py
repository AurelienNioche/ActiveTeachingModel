import numpy as np
from scipy.special import logsumexp
from itertools import product

EPS = np.finfo(np.float).eps


def post_mean_sd(log_post, grid_param):

    post_mean__ = post_mean(log_post=log_post, grid_param=grid_param)
    post_sd__ = post_sd(log_post=log_post, grid_param=grid_param,
                        post_mean__=post_mean__)
    return post_mean__, post_sd__


def post_mean(log_post, grid_param) -> np.ndarray:
    """
    A vector of estimated means for the posterior distribution.
    Its length is ``n_param_set``.
    """
    return np.dot(np.exp(log_post), grid_param)


def post_cov(grid_param, log_post, post_mean__) -> np.ndarray:
    """
    An estimated covariance matrix for the posterior distribution.
    Its shape is ``(num_grids, n_param_set)``.
    """
    # shape: (N_grids, N_param)
    # _post_mean = post_mean(log_post=log_post, grid_param=grid_param)
    d = grid_param - post_mean__
    return np.dot(d.T, d * np.exp(log_post).reshape(-1, 1))


def post_sd(grid_param, log_post, post_mean__) -> np.ndarray:
    """
    A vector of estimated standard deviations for the posterior
    distribution. Its length is ``n_param_set``.
    """
    _post_cov = post_cov(grid_param=grid_param, log_post=log_post,
                         post_mean__=post_mean__)
    return np.sqrt(np.diag(_post_cov))


class Psychologist:

    def __init__(self, n_iter, learner, grid_size=20):

        self.learner = learner

        bounds = np.array(self.learner.bounds)

        self.grid_size = grid_size
        self.grid_param = self.compute_grid_param(bounds=bounds,
                                                  grid_size=self.grid_size)
        self.n_param_set = len(self.grid_param)
        lp = np.ones(self.n_param_set)
        self.log_post = lp - logsumexp(lp)

        self.pm, self.psd = \
            post_mean_sd(grid_param=self.grid_param,
                         log_post=self.log_post)

        n_param = len(bounds)
        self.hist_pm = np.zeros((n_iter, n_param))
        self.hist_psd = np.zeros((n_iter, n_param))
        self.c_iter = 0

    @staticmethod
    def compute_grid_param(grid_size, bounds):
        return np.asarray(list(
            product(*[
                np.linspace(*b, grid_size)
                for b in bounds])))

    def update(self, item, response):

        if self.learner.n_pres[item] == 0 or self.learner.delta[item] == 0:
            pass
        else:
            log_lik = self.learner.log_lik(item=item,
                                           grid_param=self.grid_param)

            # Update prior
            self.log_post += log_lik[:, int(response)].flatten()
            self.log_post -= logsumexp(self.log_post)

            # Compute post mean and std
            self.pm, self.psd = \
                post_mean_sd(grid_param=self.grid_param,
                             log_post=self.log_post)

        # Backup
        self.hist_pm[self.c_iter] = self.pm
        self.hist_psd[self.c_iter] = self.psd

        # # Update learner
        # self.learner.update(item)

        self.c_iter += 1

        return self.pm
