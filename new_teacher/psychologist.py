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

    def __init__(self, n_item, n_iter_per_ss, n_iter_between_ss, n_ss,
                 grid_size=20, bounds=((0.001, 0.04), (0.2, 0.5)),
                 true_param=None):

        self.n_pres = np.zeros(n_item, dtype=int)
        self.delta = np.zeros(n_item, dtype=int)

        self.n_iter_per_ss = n_iter_per_ss
        self.n_iter_between_ss = n_iter_between_ss

        self.items = np.arange(n_item)
        self.n_item = n_item

        self.n_pres = np.zeros(n_item, dtype=int)
        self.delta = np.zeros(n_item, dtype=int)

        self.c_iter_session = 0
        self.c_iter = 0

        self.items = np.arange(n_item)

        n_param = len(bounds)
        n_iter = n_iter_per_ss * n_ss
        self.hist_pm = np.zeros((n_iter, n_param))
        self.hist_psd = np.zeros((n_iter, n_param))

        bounds = np.array(bounds)
        self.grid_size = grid_size
        self.grid_param = self.compute_grid_param(bounds=bounds,
                                                  grid_size=self.grid_size)
        self.n_param_set = len(self.grid_param)
        lp = np.ones(self.n_param_set)
        self.log_post = lp - logsumexp(lp)

        self.pm, self.psd = \
            post_mean_sd(grid_param=self.grid_param,
                         log_post=self.log_post)

        self.true_param = true_param

    @staticmethod
    def compute_grid_param(grid_size, bounds):
        return np.asarray(list(
            product(*[
                np.linspace(*b, grid_size)
                for b in bounds])))

    def update(self, item, response):

        if self.n_pres[item] == 0 or self.delta[item] == 0:
            pass
        else:
            log_lik = self._log_lik(item)

            if response is None:
                response = self._estimate_response(item)

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

        self.n_pres[item] += 1

        # Increment delta for all items
        self.delta[:] += 1
        # ...except the one for the selected design that equal one
        self.delta[item] = 1

        self.c_iter += 1
        self.c_iter_session += 1
        if self.c_iter_session >= self.n_iter_per_ss:
            self.delta[:] += self.n_iter_between_ss
            self.c_iter_session = 0

        return self.pm

    def _log_lik(self, item):

        p = np.zeros(self.n_param_set)

        # if self.n_pres[i] == 0:
        #     raise Exception
        #
        # elif self.delta[i] == 0:
        #     p[:] = 1
        #
        # else:
        fr = self.grid_param[:, 0] \
            * (1 - self.grid_param[:, 1]) ** (self.n_pres[item] - 1)

        p[:] = np.exp(- fr * self.delta[item])

        p_failure_success = np.zeros((self.n_param_set, 2))
        p_failure_success[:, 0] = 1 - p
        p_failure_success[:, 1] = p

        log_lik = np.log(p_failure_success + EPS)
        return log_lik

    def _estimate_response(self, item):

        # if self.n_pres[item] == 0:
        #     return 0
        #
        # elif self.delta[item] == 0:
        #     return 1
        #
        # else:
        fr = self.true_param[0] \
             * (1 - self.true_param[1]) ** (self.n_pres[item] - 1)

        p = np.exp(- fr * self.delta[item])
        r = np.random.choice([0, 1], p=[1-p, p])
        return r
