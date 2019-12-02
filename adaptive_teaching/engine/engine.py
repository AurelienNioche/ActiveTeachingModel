import numpy as np

from itertools import product

from scipy.special import logsumexp

EPS = np.finfo(np.float).eps


class Engine:

    def __init__(self, learner_model, task_param, teacher_model, teacher_param,
                 grid_size=5, gamma=1):

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
            self.teacher = teacher_model(task_param=task_param,
                                         **teacher_param)
        else:
            self.teacher = None

        self.gamma = gamma

        self.learner = learner_model(n_item=self.n_item)

    def _compute_grid_param(self, grid_size):

        return np.asarray(list(
            product(*[
                np.linspace(
                    *self.learner_model.bounds[key],
                    grid_size) for key in
                sorted(self.learner_model.bounds)])
        ))

    def _compute_log_lik(self):
        """Compute the log likelihood."""

        for i in range(self.n_item):
            self.log_lik[i, :, :] = self._log_p(i)

    def _p(self, i):

        p = self.learner.p(
            grid_param=self.grid_param,
            i=i)

        return p

    def _log_p(self, i):

        return np.log(self._p(i) + EPS)

    def _update_mutual_info(self):

        for i in range(self.n_item):

            log_p = self._log_p(i)
            self.log_lik[i, :, :] = log_p

        self.mutual_info[:] = self._mutual_info(self.log_lik,
                                                self.log_post)
        n_best = int(self.gamma*self.n_param_set)
        best_param_set_idx = \
            np.argsort(self.log_post)[-n_best:]

        for i in range(self.n_item):

            # Learn new item
            self.learner.update(i)

            log_lik_t_plus_one = np.zeros((
                self.n_item,
                n_best,
                2))

            for j in range(self.n_item):

                p = self.learner.p(
                    grid_param=self.grid_param[best_param_set_idx],
                    i=i,
                )

                new_log_p = np.log(p + EPS)

                log_lik_t_plus_one[j, :, :] = new_log_p

            mutual_info_t_plus_one_for_seq_i_j = \
                self._mutual_info(log_lik_t_plus_one,
                                  self.log_post[best_param_set_idx])
            max_info_next_time_step = \
                np.max(mutual_info_t_plus_one_for_seq_i_j)

            self.mutual_info[i] += max_info_next_time_step

            # Unlearn item
            self.learner.cancel_update()

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

        self.learner.update(item)
        self.teacher.update(item, response)

    def get_item(self):

        if np.max(self.post_sd) > self.teacher.confidence_threshold:

            self._update_mutual_info()
            return np.random.choice(
                self.items[self.mutual_info == np.max(self.mutual_info)]
            )

        else:

            self._compute_log_lik()
            return self.teacher.ask(
                best_param=self.post_mean)
