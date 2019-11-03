import numpy as np

from . revised import AdaptiveRevised

from scipy.special import logsumexp

EPS = np.finfo(np.float).eps

# from tqdm import tqdm
# from time import time
# import datetime

RANDOM = "Random"
OPT_INF0 = "Opt. info"
OPT_TEACH = "Opt. teach only"
ADAPTIVE = "Adaptive"


class TeacherHalfLife(AdaptiveRevised):

    confidence_threshold = 0.1

    def __init__(self, design_type, **kwargs):

        if design_type not in (RANDOM, OPT_TEACH, OPT_INF0, ADAPTIVE):
            raise ValueError

        super().__init__(**kwargs)

        # self.learning_value = np.zeros(self.n_design)

        self.delta = np.zeros(self.n_design, dtype=int)
        self.n_pres_minus_one = np.full(self.n_design, -1,
                                        dtype=int)
        self.seen = np.zeros(self.n_design, dtype=bool)

        self.delta_i = None
        self.seen_i = None
        self.i = None

        self.get_design = {
            OPT_INF0: self._optimize_info_selection,
            OPT_TEACH: self._optimize_teaching_selection,
            RANDOM: self._random_selection,
            ADAPTIVE: self._adaptive_selection
        }[design_type]

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

    def _compute_log_lik(self):
        """Compute the log likelihood."""

        for i in range(self.n_design):
            self.log_lik[i, :, :] = self._log_p(i)

    def _p(self, i):

        p = np.zeros((self.n_param_set, 2))

        seen = self.seen[i] == 1
        if seen:
            p[:, 1] = np.exp(
                - self.grid_param[:, 1]
                * (1 - self.grid_param[:, 0]) ** self.n_pres_minus_one[i]
                * self.delta[i])

        p[:, 0] = 1 - p[:, 1]

        return p

    def _log_p(self, i):

        return np.log(self._p(i) + EPS)

    def _update_mutual_info(self):

        # self.log_lik_seq = np.zeros((self.n_seq, self.n_param_set, 4))

        # n_seq = 0

        for i in range(self.n_design):

            log_p = self._log_p(i)
            self.log_lik[i, :, :] = log_p

        self.mutual_info[:] = self._mutual_info(self.log_lik,
                                                self.log_post)
        n_best = int(.05*self.n_param_set)
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

                p = np.zeros((n_best, 2))

                seen = self.seen[i] == 1
                if seen:
                    p[:, 1] = np.exp(
                        - self.grid_param[best_param_set_idx, 1]
                        * (1 - self.grid_param[best_param_set_idx, 0])
                        ** self.n_pres_minus_one[i]
                        * self.delta[i])

                p[:, 0] = 1 - p[:, 1]

                new_log_p = np.log(p + EPS)

                log_lik_t_plus_one[j, :, :] = new_log_p

            mutual_info_t_plus_one_for_seq_i_j = \
                self._mutual_info(log_lik_t_plus_one,
                                  self.log_post[best_param_set_idx])

            self.mutual_info[i] += \
                np.max(mutual_info_t_plus_one_for_seq_i_j)

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

    def _optimize_info_selection(self):
        self._update_mutual_info()
        return self._select_design(self.mutual_info)

    def _random_selection(self):
        self._compute_log_lik()
        return np.random.choice(self.possible_design)

    def _optimize_teaching_selection(self):

        self._compute_log_lik()
        alpha, beta = self.post_mean

        seen = self.seen[:] == 1

        if not np.any(seen):
            print("Introduce new item (not any seen)")
            return np.random.choice(self.possible_design)

        p_recall_seen = np.exp(
            - beta
            * (1 - alpha) ** self.n_pres_minus_one[seen]
            * self.delta[seen])

        u = 1-p_recall_seen

        sum_u = np.sum(u)
        if sum_u <= 1 and np.sum(seen) < len(seen) and np.random.random() < 1-sum_u:
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
