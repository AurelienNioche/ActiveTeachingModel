import numpy as np

from . teacher import Teacher

from scipy.special import logsumexp

EPS = np.finfo(np.float).eps

from tqdm import tqdm
from time import time
import datetime


class TeacherHalfLife(Teacher):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.delta = np.zeros(len(self.possible_design), dtype=int)
        self.n_pres_minus_one = np.full(len(self.possible_design), -1,
                                        dtype=int)
        self.seen = np.zeros(len(self.possible_design), dtype=bool)

        self.delta_i = None
        self.seen_i = None
        self.i = None

    def _update_learning_value(self):

        for i in range(len(self.possible_design)):

            self._update_history(i)

            v = np.zeros(len(self.possible_design))

            for j in range(len(self.possible_design)):

                v[j] = logsumexp(self.log_post[:] + self._log_p(j)[:, 1])

            self.learning_value[i] = np.sum(v)

            self._cancel_last_update_history()

    def _compute_log_lik(self, one_step_forward=False):
        """Compute the log likelihood."""

        # t = time()
        # tqdm.write("Compute log")

        for i in range(len(self.possible_design)):

            log_p = self._log_p(i)

            self.log_lik[i, :, :] = log_p

            if one_step_forward:

                # Learn new item
                self._update_history(i)

                new_log_p = self._log_p(i)

                self.log_lik_t0_t1[i, :, :] = np.hstack((log_p, new_log_p))

                # Unlearn item
                self._cancel_last_update_history()

        # tqdm.write(f"Done! [time elapsed "
        # f"{datetime.timedelta(seconds=time() - t)}]")

    def _p(self, i):

        p = np.zeros((len(self.grid_param), 2))

        seen = self.seen[i] == 1
        if seen:
            p[:, 1] = np.exp(
                - self.grid_param[:, 1]
                * (1 - self.grid_param[:, 0]) ** self.n_pres_minus_one[i]
                * self.delta[i])
        else:
            p[:, 1] = np.zeros(len(self.grid_param))

        p[:, 0] = 1 - p[:, 1]

        return p

    def _log_p(self, i):

        return np.log(self._p(i) + EPS)

    def _update_history(self, design):

        self.i = design
        self.delta_i = self.delta[design]
        self.seen_i = self.seen[design]

        self.n_pres_minus_one[design] += 1
        self.delta += 1
        self.delta[design] = 1
        self.seen[design] = 1

    def _cancel_last_update_history(self):

        self.n_pres_minus_one[self.i] -= 1
        self.delta -= 1
        self.delta[self.i] = self.delta_i
        self.seen[self.i] = self.seen_i

    def get_design(self, kind='optimal'):
        # type: (str) -> int
        r"""
        Choose a design with a given type.

        * ``optimal``: an optimal design :math:`d^*` that maximizes the mutual
          information.
        * ``random``: a design randomly chosen.

        Parameters
        ----------
        kind : {'optimal', 'random'}, optional
            Type of a design to choose

        Returns
        -------
        design : int
            A chosen design
        """

        if kind == 'optimal':
            self._compute_log_lik(one_step_forward=True)
            self._update_mutual_info()
            design = self._select_design(self.mutual_info)

        elif kind == 'random':
            self._compute_log_lik()
            design = np.random.choice(self.possible_design)

        elif kind == 'pure_teaching':
            self._compute_log_lik()
            self._update_learning_value()
            design = self._select_design(self.learning_value)

        elif kind == 'adaptive_teaching':
            if np.max(self.post_sd) > 0.1:
                self._compute_log_lik(one_step_forward=True)
                self._update_mutual_info()
                design = self._select_design(self.mutual_info)
            else:
                self._compute_log_lik()
                self._update_learning_value()
                design = self._select_design(self.learning_value)

        else:
            raise ValueError(
                'The argument kind is wrong.')
        return design
