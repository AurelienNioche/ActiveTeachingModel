import numpy as np
from scipy.special import logsumexp
from itertools import product
from abc import abstractmethod

EPS = np.finfo(np.float).eps


# def post_mean_sd(log_post, grid_param):
#
#     post_mean__ = post_mean(log_post=log_post, grid_param=grid_param)
#     post_sd__ = post_sd(log_post=log_post, grid_param=grid_param,
#                         post_mean__=post_mean__)
#     return post_mean__, post_sd__


class Psychologist:

    def __init__(self, learner, n_iter, grid_size):
        self.n_iter = n_iter
        self.learner = learner

        self.grid_size = grid_size
        self.grid_param = self.cp_grid_param(bounds=self.learner.bounds,
                                             grid_size=self.grid_size)
        self.n_param_set = len(self.grid_param)

        self.hist_pm = None
        self.hist_psd = None

    @staticmethod
    def cp_grid_param(grid_size, bounds):
        return np.asarray(list(
            product(*[
                np.linspace(*b, grid_size)
                for b in bounds])))

    @staticmethod
    def cp_post_mean(log_post, grid_param) -> np.ndarray:
        """
        A vector of estimated means for the posterior distribution.
        Its length is ``n_param_set``.
        """
        return np.dot(np.exp(log_post), grid_param)

    @staticmethod
    def cp_post_cov(grid_param, log_post, post_mean) -> np.ndarray:
        """
        An estimated covariance matrix for the posterior distribution.
        Its shape is ``(num_grids, n_param_set)``.
        """
        # shape: (N_grids, N_param)
        # _post_mean = post_mean(log_post=log_post, grid_param=grid_param)
        d = grid_param - post_mean
        return np.dot(d.T, d * np.exp(log_post).reshape(-1, 1))

    @classmethod
    def cp_post_sd(cls, grid_param, log_post, post_mean) -> np.ndarray:
        """
        A vector of estimated standard deviations for the posterior
        distribution. Its length is ``n_param_set``.
        """
        _post_cov = cls.cp_post_cov(grid_param=grid_param, log_post=log_post,
                                    post_mean=post_mean)
        return np.sqrt(np.diag(_post_cov))

    @classmethod
    def get(cls, learner, n_iter, grid_size=20):
        if learner.heterogeneous_param:
            return PsychologistHeterogeneous(
                learner=learner,
                n_iter=n_iter,
                grid_size=grid_size)
        else:
            return PsychologistHomogeneous(
                learner=learner,
                n_iter=n_iter,
                grid_size=grid_size)

    @abstractmethod
    def update(self, item, response):
        return


class PsychologistHomogeneous(Psychologist):

    def __init__(self, n_iter, learner, grid_size=20):

        super().__init__(learner=learner, n_iter=n_iter, grid_size=grid_size)

        lp = np.ones(self.n_param_set)
        self.log_post = lp - logsumexp(lp)

        self.post_mean = self.cp_post_mean(grid_param=self.grid_param,
                                           log_post=self.log_post)
        self.post_std = self.cp_post_sd(grid_param=self.grid_param,
                                        log_post=self.log_post,
                                        post_mean=self.post_mean)

        n_param = len(self.learner.bounds)
        self.hist_pm = np.zeros((n_iter, n_param))
        self.hist_psd = np.zeros((n_iter, n_param))
        self.c_iter = 0

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
            self.post_mean = self.cp_post_mean(grid_param=self.grid_param,
                                               log_post=self.log_post)
            self.post_std = self.cp_post_sd(grid_param=self.grid_param,
                                            log_post=self.log_post,
                                            post_mean=self.post_mean)

        # Backup
        self.hist_pm[self.c_iter] = self.post_mean
        self.hist_psd[self.c_iter] = self.post_std

        self.c_iter += 1


class PsychologistHeterogeneous(Psychologist):

    def __init__(self, learner, n_iter, grid_size=20):

        super().__init__(learner=learner, n_iter=n_iter, grid_size=grid_size)

        n_item = learner.n_item

        lp = np.ones(self.n_param_set)
        lp -= logsumexp(lp)
        self.log_post = np.tile(lp, (n_item, 1))

        pm = self.cp_post_mean(log_post=lp, grid_param=self.grid_param)
        self.post_mean = np.tile(pm, (n_item, 1))

        # n_param = len(bounds)
        # self.hist_pm = np.zeros((n_iter, n_param))
        # self.hist_psd = np.zeros((n_iter, n_param))
        # self.c_iter = 0

        self.hist_pm = [[] for _ in range(self.learner.n_item)]

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
            lp = self.log_post[item]

            # Update prior
            lp += log_lik[:, int(response)].flatten()
            lp -= logsumexp(self.log_post)

            # Compute post mean and std
            pm = self.cp_post_mean(grid_param=self.grid_param, log_post=lp)

            self.post_mean[item] = pm
            self.log_post[item] = lp

            self.hist_pm[item].append((self.learner.c_iter, *pm))
