import numpy as np
from itertools import product
from scipy.special import logsumexp
# from deco import concurrent, synchronized

EPS = np.finfo(np.float).eps

from multiprocessing import Pool

# np.seterr(all='raise')


class Engine:

    def __init__(self, learner_model, task_param, teacher_model, teacher_param,
                 grid_size=5, true_param=None):

        self.learner_model = learner_model

        self.grid_param = self._compute_grid_param(grid_size)
        self.n_param_set = len(self.grid_param)

        self.n_item = task_param['n_item']
        self.items = np.arange(self.n_item)

        self.log_lik = np.zeros((self.n_item,
                                 self.n_param_set, 2))

        self.y = np.arange(2)

        self.hist = []
        self.responses = []

        # Post <= prior
        # shape (n_param_set, )
        lp = np.ones(self.n_param_set)
        self.log_post = lp - logsumexp(lp)

        self.mutual_info = np.zeros(self.n_item)

        if teacher_model is not None:
            print(teacher_model)
            self.teacher = teacher_model(task_param=task_param,
                                         **teacher_param)
        else:
            self.teacher = None

        # self.gamma = gamma

        self.learner = learner_model(task_param=task_param)

        self.true_param = true_param

        self.ll_after_pres_full = \
            np.zeros((self.n_item, self.n_param_set, 2))
        self.ll_without_pres_full = \
            np.zeros((self.n_item, self.n_param_set, 2))

        self.t = 0

    def _compute_grid_param(self, grid_size):

        return np.asarray(list(
            product(*[
                np.linspace(
                    *self.learner_model.bounds[key],
                    grid_size) for key in
                sorted(self.learner_model.bounds)])
        ))

    def _compute_log_lik(self):
        for i in range(self.n_item):
            self.log_lik[i, :, :] = self._log_p(i)

    def _log_p(self, i):

        p = self.learner.p_grid(grid_param=self.grid_param, i=i)
        log = np.log(p + EPS)
        return log

    def _update_mutual_info(self):
        from time import time

        n_seen = int(np.sum(self.learner.seen))
        n_not_seen = self.n_item - n_seen

        seen = np.zeros(self.n_item, dtype=bool)
        seen[:] = self.learner.seen

        not_seen = np.zeros(self.n_item, dtype=bool)
        not_seen[:] = np.logical_not(self.learner.seen)

        if n_seen == 0:
            self._compute_log_lik()
            return

        n_sample = min(n_seen+1, self.n_item)

        log_lik = np.zeros((n_sample, self.n_param_set, 2))

        items_seen = self.items[self.learner.seen]

        if n_not_seen > 0:
            item_not_seen = self.items[not_seen][0]
            item_sample = list(items_seen) + [item_not_seen, ]
        else:
            item_sample = items_seen

        for i, item in enumerate(item_sample):
            log_lik[i, :, :] = self._log_p(item)

        self.log_lik[seen] = log_lik[:n_seen]
        if n_not_seen:
            self.log_lik[not_seen] = log_lik[-1]

        self.mutual_info[:] = self._mutual_info(
            self.log_lik,
            self.log_post)

        ll_after_pres = np.zeros((n_sample, self.n_param_set, 2))

        for i, item in enumerate(item_sample):

            for response in (0, 1):
                self.learner.update(item=item, response=response)

                ll_after_pres[i] += self._log_p(item)

                # Unlearn item
                self.learner.cancel_update()

        ll_without_pres = np.zeros((n_sample, self.n_param_set, 2))

        # Go to the future
        self.learner.update(item=None, response=None)

        for i, item in enumerate(item_sample):
            ll_without_pres[i] = self._log_p(item)

        # Cancel
        self.learner.cancel_update()

        self.ll_after_pres_full[seen] = ll_after_pres[:n_seen]

        self.ll_without_pres_full[seen] = ll_without_pres[:n_seen]

        if n_not_seen:
            # print(f"hey! {ll_after_pres[-1]}, {ll_without_pres[-1]}")
            self.ll_after_pres_full[not_seen] = ll_after_pres[-1]
            self.ll_without_pres_full[not_seen] = ll_without_pres[-1]

        max_info_next_time_step = np.zeros(self.n_item)

        with Pool() as pool:
            max_info_next_time_step[:] = \
                pool.map(self._compute_max_info_time_step, self.items)

        self.mutual_info += max_info_next_time_step

    def _compute_max_info_time_step(self, i):

        ll_t_plus_one = np.zeros((self.n_item, self.n_param_set, 2))

        ll_t_plus_one[:] = self.ll_without_pres_full[:]
        ll_t_plus_one[i] = self.ll_after_pres_full[i]

        mutual_info_t_plus_one_given_i = \
            self._mutual_info(ll_t_plus_one,
                              self.log_post)

        return np.max(mutual_info_t_plus_one_given_i)


    # def _update_mutual_info(self):
    #
    #     for i in range(self.n_item):
    #         self.log_lik[i, :, :] = self._log_p(i)
    #
    #     self.mutual_info[:] = self._mutual_info(self.log_lik,
    #                                             self.log_post)
    #
    #     for i in range(self.n_item):
    #
    #         # Learn new item
    #         self.learner.update(item=i, response=None)
    #
    #         log_lik_t_plus_one = np.zeros((
    #             self.n_item,
    #             self.n_param_set,  # n_best,
    #             2))
    #
    #         for j in range(self.n_item):
    #             log_lik_t_plus_one[j, :, :] = self._log_p(j)
    #
    #         mutual_info_t_plus_one_given_i = \
    #             self._mutual_info(log_lik_t_plus_one,
    #                               self.log_post)
    #
    #         max_info_next_time_step = \
    #             np.max(mutual_info_t_plus_one_given_i)
    #
    #         self.mutual_info[i] += max_info_next_time_step
    #
    #         # Unlearn item
    #         self.learner.cancel_update()

    # def _update_mutual_info_asymmetric(self):
    #
    #         for i in range(self.n_item):
    #             self.log_lik[i, :, :] = self._log_p(i)
    #
    #         self.mutual_info[:] = self._mutual_info(self.log_lik,
    #                                                 self.log_post)
    #         # n_best = int(self.gamma*self.n_param_set)
    #         # best_param_set_idx = \
    #         #     np.argsort(self.log_post)[-n_best:]
    #         print("t", self.t, "=" * 10)
    #         print("mutual info t only", self.mutual_info)
    #
    #         for i in range(self.n_item):
    #
    #             self.learner.set_param(self.post_mean)
    #             p_success = self.learner.p(i)
    #
    #             mutual_info_t_plus_one_for_seq_i_j = np.zeros((2, self.n_item))
    #
    #             for response in (0, 1):
    #                 # Learn new item
    #                 self.learner.update(item=i, response=response)
    #
    #                 log_lik_t_plus_one = np.zeros((
    #                     self.n_item,
    #                     self.n_param_set,  # n_best,
    #                     2))
    #
    #                 for j in range(self.n_item):
    #                     log_lik_t_plus_one[j, :, :] = self._log_p(j)
    #
    #                 mutual_info_t_plus_one_for_seq_i_j[response] = \
    #                     self._mutual_info(log_lik_t_plus_one,
    #                                       self.log_post)
    #                 # self.log_post[best_param_set_idx])
    #
    #                 # Unlearn item
    #                 self.learner.cancel_update()
    #
    #             max_info_next_time_step = \
    #                 np.max(
    #                     mutual_info_t_plus_one_for_seq_i_j[0]*(1-p_success)
    #                     + mutual_info_t_plus_one_for_seq_i_j[1]*p_success
    #                 )
    #
    #             self.mutual_info[i] += max_info_next_time_step

    @staticmethod
    def _mutual_info(ll, lp):

        lp_reshaped = lp.reshape((1, len(lp), 1))

        # ll => likelihood
        # shape (n_item, n_param_set, num_responses, )

        # Calculate the marginal log likelihood.
        # shape (n_item, num_responses, )
        mll = logsumexp(ll + lp_reshaped, axis=1)

        # Calculate the marginal entropy and conditional entropy.
        # shape (n_item,)
        ent_mrg = - np.sum(np.exp(mll) * mll, -1)

        # Compute entropy obs -------------------------

        # shape (n_item, n_param_set, num_responses, )
        # shape (n_item, n_param_set, )
        ent_obs = - np.multiply(np.exp(ll), ll).sum(-1)

        # Compute conditional entropy -----------------

        # shape (n_item,)
        ent_cond = np.sum(np.exp(lp) * ent_obs, axis=1)

        # Calculate the mutual information. -----------
        # shape (n_item,)
        mutual_info = ent_mrg - ent_cond

        return mutual_info

    @property
    def post(self) -> np.ndarray:
        """Posterior distributions of joint parameter space"""
        return np.exp(self.log_post)

    @property
    def post_mean(self) -> np.ndarray:
        """
        A vector of estimated means for the posterior distribution.
        Its length is ``n_param_set``.
        """
        return np.dot(self.post, self.grid_param)

    @property
    def post_cov(self) -> np.ndarray:
        """
        An estimated covariance matrix for the posterior distribution.
        Its shape is ``(num_grids, n_param_set)``.
        """
        # shape: (N_grids, N_param)
        d = self.grid_param - self.post_mean
        return np.dot(d.T, d * self.post.reshape(-1, 1))

    @property
    def post_sd(self) -> np.ndarray:
        """
        A vector of estimated standard deviations for the posterior
        distribution. Its length is ``n_param_set``.
        """
        return np.sqrt(np.diag(self.post_cov))

    def update(self, item, response):
        r"""
        Update the posterior :math:`p(\theta | y_\text{obs}(t), d^*)` for
        all discretized values of :math:`\theta`.

        .. math::
            p(\theta | y_\text{obs}(t), d^*) =
                \frac{ p( y_\text{obs}(t) | \theta, d^*) p_t(\theta) }
                    { p( y_\text{obs}(t) | d^* ) }

        Parameters
        ----------
        item
            integer (0...N-1)
        response
            0 or 1
        """

        self.log_post += self.log_lik[item, :, int(response)].flatten()
        self.log_post -= logsumexp(self.log_post)

        self.learner.update(item, response)
        self.teacher.update(item, response)

    def get_item(self):

        if self.true_param is None:

            if np.max(self.post_sd) > self.teacher.confidence_threshold:

                self._update_mutual_info()
                item = np.random.choice(
                    self.items[self.mutual_info == np.max(self.mutual_info)]
                )
                # print('selected item', item)
                self.t += 1
                return item

            else:

                self._compute_log_lik()
                return self.teacher.ask(
                    best_param=self.post_mean)

        else:
            return self.teacher.ask(best_param=self.true_param)
