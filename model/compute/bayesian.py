import numpy as np


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
