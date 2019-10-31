import numpy as np

from scipy.special import logsumexp
from itertools import product


EPS = np.finfo(np.float).eps


class AdaptiveClassic:

    def __init__(self, learner_model, possible_design, grid_size=5):

        self.learner_model = learner_model
        self.n_param = len(self.learner_model.bounds)
        self.possible_design = possible_design

        self.grid_param = self._compute_grid_param(grid_size)

        self.log_lik = np.zeros((len(self.possible_design),
                                 len(self.grid_param), 2))

        self.y = np.arange(2)

        self.hist = []

        # Post <= prior
        # shape (num_params, )
        lp = np.ones(len(self.grid_param))
        self.log_post = lp - logsumexp(lp)

        self.mutual_info = None

    def _compute_grid_param(self, grid_size):

        return np.asarray(list(
            product(*[
                np.linspace(
                    *self.learner_model.bounds[key],
                    grid_size) for key in
                sorted(self.learner_model.bounds)])
        ))

    def reset(self):
        """
        Reset the engine as in the initial state.
        """

        # Reset the history
        self.hist = []

        # Reset the post / prior
        lp = np.ones(len(self.grid_param))
        self.log_post = lp - logsumexp(lp)

    @staticmethod
    def log_lik_bernoulli(y, p):
        """Log likelihood for a Bernoulli random variable"""
        return y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS)

    def _compute_log_lik(self):
        """Compute the log likelihood."""

        for j, param in enumerate(self.grid_param):

            learner = self.learner_model(
                t=len(self.hist),
                hist=self.hist,
                param=param)

            for i, x in enumerate(self.possible_design):

                p = learner.p_recall(item=x)
                log_p = self.log_lik_bernoulli(self.y, p)
                self.log_lik[i, j, :] = log_p

    def _update_mutual_info(self):

        # Get likelihood
        # shape (num_designs, num_params, num_responses, )
        ll = self.log_lik

        # Calculate the marginal log likelihood.
        # shape (num_designs, num_responses, )
        lp = self.log_post.reshape((1, len(self.log_post), 1))
        mll = logsumexp(ll + lp, axis=1)

        # Calculate the marginal entropy and conditional entropy.
        # shape (num_designs,)
        ent_mrg = - np.sum(np.exp(mll) * mll, -1)

        # Compute entropy obs -------------------------
        # shape (num_designs, num_params, )
        ent_obs = - np.multiply(np.exp(ll), ll).sum(-1)

        # Compute conditional entropy -----------------

        # shape (num_designs,)
        ent_cond = np.sum(np.exp(self.log_post) * ent_obs, axis=1)

        # Calculate the mutual information. -----------
        # shape (num_designs,)
        self.mutual_info = ent_mrg - ent_cond

    def _select_design(self, v):

        return np.random.choice(
            self.possible_design[v == np.max(v)]
        )

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

        self._compute_log_lik()

        if kind == 'optimal':
            self._update_mutual_info()
            design = self._select_design(self.mutual_info)

        elif kind == 'random':
            design = np.random.choice(self.possible_design)

        else:
            raise ValueError(
                'The argument kind should be "optimal" or "random".')
        return design

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

        idx_design = list(self.possible_design).index(design)

        self.log_post += self.log_lik[idx_design, :, response].flatten()
        self.log_post -= logsumexp(self.log_post)

        self._update_history(design)

    def _update_history(self, design):
        self.hist.append(design)

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
