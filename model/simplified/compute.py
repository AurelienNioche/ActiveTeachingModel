from itertools import product

import numpy as np


def compute_grid_param(grid_size, bounds):
    return np.asarray(list(
        product(*[
            np.linspace(*b, grid_size)
            for b in bounds])))


def post_mean(log_post, grid_param) -> np.ndarray:
    """
    A vector of estimated means for the posterior distribution.
    Its length is ``n_param_set``.
    """
    return np.dot(np.exp(log_post), grid_param)


def post_cov(grid_param, log_post) -> np.ndarray:
    """
    An estimated covariance matrix for the posterior distribution.
    Its shape is ``(num_grids, n_param_set)``.
    """
    # shape: (N_grids, N_param)
    _post_mean = post_mean(log_post=log_post, grid_param=grid_param)
    d = grid_param - _post_mean
    return np.dot(d.T, d * np.exp(log_post).reshape(-1, 1))


def post_sd(grid_param, log_post) -> np.ndarray:
    """
    A vector of estimated standard deviations for the posterior
    distribution. Its length is ``n_param_set``.
    """
    _post_cov = post_cov(grid_param=grid_param, log_post=log_post)
    return np.sqrt(np.diag(_post_cov))