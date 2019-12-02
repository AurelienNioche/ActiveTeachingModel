import numpy as np

from itertools import product

from scipy.special import logsumexp

from adaptive_design.teacher.leitner import Leitner

EPS = np.finfo(np.float).eps

# from tqdm import tqdm
# from time import time
# import datetime

RANDOM = "Random"
OPT_INF0 = "Opt. info"
OPT_TEACH = "Opt. teach only"
ADAPTIVE = "Adaptive"
LEITNER = "Leitner"


class TeacherHalfLife:

    confidence_threshold = 0.1
    gamma = 1

    def __init__(self, design_type, learner_model, possible_design,
                 grid_size=5):

        if design_type not in (RANDOM, OPT_TEACH, LEITNER,
                               OPT_INF0, ADAPTIVE):
            raise ValueError("Design type not recognized")

        self.learner_model = learner_model
        self.possible_design = possible_design

        self.grid_param = self._compute_grid_param(grid_size)

        self.n_design = len(self.possible_design)
        self.n_param_set = len(self.grid_param)

        self.log_lik = np.zeros((self.n_design,
                                 self.n_param_set, 2))

        self.y = np.arange(2)

        self.hist = []
        self.responses = []

        # Post <= prior
        # shape (num_params, )
        lp = np.ones(self.n_param_set)
        self.log_post = lp - logsumexp(lp)

        self.mutual_info = np.zeros(self.n_design)

        self.delta = np.zeros(self.n_design, dtype=int)
        self.n_pres_minus_one = np.full(self.n_design, -1,
                                        dtype=int)
        self.seen = np.zeros(self.n_design, dtype=bool)

        self.delta_i = None
        self.seen_i = None
        self.i = None

        self.design_type = design_type
        self.get_design = {
            OPT_INF0: self._optimize_info_selection,
            OPT_TEACH: self._optimize_teaching_selection,
            RANDOM: self._random_selection,
            ADAPTIVE: self._adaptive_selection,
            LEITNER: self._leitner_selection,
        }[self.design_type]

        if self.design_type == LEITNER:
            self.teacher = Leitner(n_item=self.n_design)

    # def _update_learning_value(self):
    #
    #     for i in range(self.n_design):
    #         self.log_lik[i, :, :] = self._log_p(i)
    #
    #         self._update_history(i)
    #
    #         v = np.zeros(self.n_design)
    #
    #         for j in range(self.n_design):
    #
    #             v[j] = logsumexp(self.log_post[:] + self._log_p(j)[:, 1])
    #
    #         self.learning_value[i] = np.sum(v)
    #
    #         self._cancel_last_update_history()

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

        for i in range(self.n_design):
            self.log_lik[i, :, :] = self._log_p(i)

    def _p(self, i):

        p = self.learner_model.p(
            grid_param=self.grid_param,
            n_pres_minus_one=self.n_pres_minus_one,
            seen=self.seen,
            delta=self.delta,
            i=i)

        return p

    def _log_p(self, i):

        return np.log(self._p(i) + EPS)

    def _update_mutual_info(self):

        for i in range(self.n_design):

            log_p = self._log_p(i)
            self.log_lik[i, :, :] = log_p

        self.mutual_info[:] = self._mutual_info(self.log_lik,
                                                self.log_post)
        n_best = int(self.gamma*self.n_param_set)
        best_param_set_idx = \
            np.argsort(self.log_post)[-n_best:]

        for i in range(self.n_design):

            # Learn new item
            self._update_history(i)

            log_lik_t_plus_one = np.zeros((
                self.n_design,
                n_best,
                2))

            for j in range(self.n_design):

                p = self.learner_model.p(
                    grid_param=self.grid_param[best_param_set_idx],
                    n_pres_minus_one=self.n_pres_minus_one,
                    seen=self.seen[i],
                    delta=self.delta[i],
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
            self._cancel_last_update_history()

    @staticmethod
    def _mutual_info(ll, lp):

        lp_reshaped = lp.reshape((1, len(lp), 1))

        # ll => likelihood
        # shape (num_designs, num_params, num_responses, )

        # Calculate the marginal log likelihood.
        # shape (num_designs, num_responses, )
        mll = logsumexp(ll + lp_reshaped, axis=1)

        # Calculate the marginal entropy and conditional entropy.
        # shape (num_designs,)
        ent_mrg = - np.sum(np.exp(mll) * mll, -1)

        # Compute entropy obs -------------------------

        # shape (num_designs, num_params, num_responses, )
        # shape (num_designs, num_params, )
        ent_obs = - np.multiply(np.exp(ll), ll).sum(-1)

        # Compute conditional entropy -----------------

        # shape (num_designs,)
        ent_cond = np.sum(np.exp(lp) * ent_obs, axis=1)

        # Calculate the mutual information. -----------
        # shape (num_designs,)
        mutual_info = ent_mrg - ent_cond

        return mutual_info

    def _select_design(self, v):

        return np.random.choice(
            self.possible_design[v == np.max(v)]
        )

    def _optimize_info_selection(self):
        self._update_mutual_info()
        return self._select_design(self.mutual_info)

    def _random_selection(self):
        self._compute_log_lik()
        return np.random.choice(self.possible_design)

    def _leitner_selection(self):
        self._compute_log_lik()
        item = self.teacher.ask(
            t=len(self.hist),
            hist_success=self.responses,
            hist_item=self.hist,
        )
        return item

    def _optimize_teaching_selection(self):

        self._compute_log_lik()
        best_param = self.post_mean

        seen = self.seen[:] == 1

        if not np.any(seen):
            return np.random.choice(self.possible_design)

        p_recall_seen = self.learner_model.p_recall_seen(
            n_pres_minus_one=self.n_pres_minus_one,
            delta=self.delta, seen=seen, param=best_param
        )

        u = 1-p_recall_seen

        sum_u = np.sum(u)
        if sum_u <= 1 \
                and np.sum(seen) < len(seen) \
                and np.random.random() < 1-sum_u:
            return np.random.choice(self.possible_design[np.invert(seen)])

        else:
            u /= np.sum(u)
            return np.random.choice(self.possible_design[seen], p=u)

        # self._update_learning_value()
        # return self._select_design(self.learning_value)

    def _adaptive_selection(self):
        if np.max(self.post_sd) > self.confidence_threshold:
            return self._optimize_info_selection()
        else:
            return self._optimize_teaching_selection()

    @property
    def post(self) -> np.ndarray:
        """Posterior distributions of joint parameter space"""
        return np.exp(self.log_post)

    @property
    def post_mean(self) -> np.ndarray:
        """
        A vector of estimated means for the posterior distribution.
        Its length is ``num_params``.
        """
        return np.dot(self.post, self.grid_param)

    @property
    def post_cov(self) -> np.ndarray:
        """
        An estimated covariance matrix for the posterior distribution.
        Its shape is ``(num_grids, num_params)``.
        """
        # shape: (N_grids, N_param)
        d = self.grid_param - self.post_mean
        return np.dot(d.T, d * self.post.reshape(-1, 1))

    @property
    def post_sd(self) -> np.ndarray:
        """
        A vector of estimated standard deviations for the posterior
        distribution. Its length is ``num_params``.
        """
        return np.sqrt(np.diag(self.post_cov))

    def _update_history(self, design):

        self.i = design
        self.delta_i = self.delta[design]
        self.seen_i = self.seen[design]

        # Increment delta for all items
        self.delta += 1

        # ...except the one for the selected design that equal one
        self.delta[design] = 1
        self.seen[design] = 1
        self.n_pres_minus_one[design] += 1

    def _cancel_last_update_history(self):

        self.n_pres_minus_one[self.i] -= 1
        self.delta -= 1
        self.delta[self.i] = self.delta_i
        self.seen[self.i] = self.seen_i

    def update(self, design, response):
        r"""
        Update the posterior :math:`p(\theta | y_\text{obs}(t), d^*)` for
        all discretized values of :math:`\theta`.

        .. math::
            p(\theta | y_\text{obs}(t), d^*) =
                \frac{ p( y_\text{obs}(t) | \theta, d^*) p_t(\theta) }
                    { p( y_\text{obs}(t) | d^* ) }

        Parameters
        ----------
        design
            Design vector for given response
        response
            0 or 1
        """

        print("design type", self.design_type,
              "item", design,
              "response", response)

        idx_design = list(self.possible_design).index(design)

        self.log_post += self.log_lik[idx_design, :, response].flatten()
        self.log_post -= logsumexp(self.log_post)

        self._update_history(design)
        self.responses.append(response)
        self.hist.append(design)
