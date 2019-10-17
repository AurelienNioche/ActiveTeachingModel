import os

import numpy as np
import matplotlib.pyplot as plt

# from adopy import Engine, Model, Task

from tqdm import tqdm

from learner.half_life import HalfLife

from scipy.special import logsumexp

from itertools import product

EPS = np.finfo(np.float).eps


class Adaptive:

    def __init__(self, learner_model, possible_design, grid_size=5):

        self.learner_model = learner_model

        # n param
        self.n_param = len(self.learner_model.bounds)

        self.grid_size = grid_size

        self.possible_design = possible_design

        self._compute_grid_param()
        self.reset()

    def _compute_grid_param(self):

        self.grid_param = np.asarray(list(
            product(*[
                np.linspace(
                    *self.learner_model.bounds[key],
                    self.grid_size) for key in
                sorted(self.learner_model.bounds)])
        ))

    def reset(self):
        """
        Reset the engine as in the initial state.
        """

        self.hist = []

        self.log_lik = np.zeros((len(self.possible_design),
                                 len(self.grid_param), 2))

        self._compute_log_lik()

        lp = np.ones(len(self.grid_param))
        self.log_prior = lp - logsumexp(lp)
        self.log_post = self.log_prior.copy()

        self._compute_marg_log_lik()

        ll = self.log_lik

        self.ent_obs = -np.multiply(np.exp(ll), ll).sum(-1)
        self.ent_marg = None
        self.ent_cond = None
        self.mutual_info = None

        self.flag_update_mutual_info = True

    def _p_obs(self, item, param):

        hist = self.hist + [item, ]

        learner = self.learner_model(
            hist=hist,
            param=param)
        p_obs = learner.p_recall(item=item)
        return p_obs

    def _compute_log_lik(self):
        """Compute the log likelihood."""

        for i, x in enumerate(self.possible_design):
            for j, param in enumerate(self.grid_param):
                p = self._p_obs(x, param)
                for y in (0, 1):
                    self.log_lik[i, j, y] = y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS)

    def _compute_marg_log_lik(self):

        """Compute the marginal log likelihood"""

        # Calculate the marginal log likelihood.
        lp = self.log_post.reshape((1, len(self.log_post), 1))
        mll = logsumexp(self.log_lik + lp, axis=1)
        self.marg_log_lik = mll  # shape (num_design, num_response)

    def _update_mutual_info(self):

        # If there is no need to update mutual information, it ends.
        if not self.flag_update_mutual_info:
            return

        # Calculate the marginal log likelihood.
        self._compute_marg_log_lik()

        # Calculate the marginal entropy and conditional entropy.
        self.ent_marg = -np.sum(
            np.exp(self.marg_log_lik)
            * self.marg_log_lik, -1)  # shape (num_designs,)

        self.ent_cond = np.sum(
            self.post * self.ent_obs, axis=1)  # shape (num_designs,)

        # Calculate the mutual information.
        self.mutual_info = self.ent_marg - \
            self.ent_cond  # shape (num_designs,)

        # Flag that there is no need to update mutual information again.
        self.flag_update_mutual_info = False

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
            self._update_mutual_info()
            idx_design = np.argmax(self.mutual_info)
            design = self.possible_design[idx_design]
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

        self.hist.append(design)

        idx_design = list(self.possible_design).index(design)

        self.log_post += self.log_lik[idx_design, :, response].flatten()
        self.log_post -= logsumexp(self.log_post)

        self.flag_update_mutual_info = True

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


def create_fig(param, design_types, post_means, post_sds, true_param,
               num_trial):

    fig, axes = plt.subplots(ncols=len(param), figsize=(12, 6))

    colors = [f'C{i}' for i in range(len(param))]

    for i, ax in enumerate(axes):

        for j, dt in enumerate(design_types):

            pr = param[i]

            means = post_means[pr][dt]
            stds = post_sds[pr][dt]

            true_p = true_param[pr]
            ax.axhline(true_p, linestyle='--', color='black',
                       alpha=.2)

            ax.plot(means, color=colors[j], label=dt)
            ax.fill_between(range(num_trial), means-stds,
                            means+stds, alpha=.2, color=colors[j])

            ax.set_title(pr)

    plt.legend()
    FIG_FOLDER = os.path.join("fig", "adaptive")
    os.makedirs(FIG_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(FIG_FOLDER, "adaptive.pdf"))


class FakeModel:

    bounds = {
        "b0": (0, 10),
        "b1": (0, 10)
    }

    def __init__(self, param, hist=None):

        if isinstance(param, dict):
            self.b0 = param['b0']
            self.b1 = param['b1']
        else:
            self.b0, self.b1 = param

        self.hist = None

    def p_recall(self, item):
        """A function to compute the probability of a positive response."""

        logit = self.b0 + item * self.b1**2
        p_obs = 1. / (1 + np.exp(-logit))

        return p_obs

    def learn(self, **kwargs):
        pass


def main():

    print("Preparing...")

    np.random.seed(123)

    grid_size = 100

    possible_design = np.arange(10)
    learner_model = HalfLife
    true_param = {
        "beta": 0.02,
        "alpha": 0.2
    }

    # possible_design = np.linspace(0, 10, 100)
    # learner_model = FakeModel
    # true_param = {
    #     "b0": 2,
    #     "b1": 3,
    # }

    num_trial = 200

    param = sorted(learner_model.bounds.keys())

    learner = learner_model(param=true_param)
    engine = Adaptive(learner_model=learner_model,
                      possible_design=possible_design,
                      grid_size=grid_size)

    design_types = ['optimal', 'random']

    post_means = {pr: {d: np.zeros(num_trial)
                       for d in design_types}
                  for pr in param}
    post_sds = {pr: {d: np.zeros(num_trial)
                     for d in design_types}
                for pr in param}

    # Run simulations for three designs
    for design_type in design_types:

        print("Resetting the engine...")

        # Reset the engine as an initial state
        engine.reset()

        print(f"Computing results for design '{design_type}'...")

        for trial in tqdm(range(num_trial)):

            # Compute an optimal design for the current trial
            design = engine.get_design(design_type)

            # Get a response using the optimal design
            p = learner.p_recall(item=design)
            response = int(p > np.random.random())

            # Update the engine
            engine.update(design, response)

            for i, pr in enumerate(param):
                post_means[pr][design_type][trial] = engine.post_mean[i]
                post_sds[pr][design_type][trial] = engine.post_sd[i]

    create_fig(param=param, design_types=design_types,
               post_means=post_means, post_sds=post_sds,
               true_param=true_param, num_trial=num_trial)


if __name__ == '__main__':
    main()
