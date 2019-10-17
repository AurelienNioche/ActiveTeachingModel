import os

import numpy as np
import matplotlib.pyplot as plt

# from adopy import Engine, Model, Task

from tqdm import tqdm

from learner.half_life import HalfLife

from scipy.special import logsumexp

from itertools import product


class Adaptive:

    def __init__(self, learner_model, possible_design, grid_size=5):

        self.learner_model = learner_model

        # n param
        self.n_param = len(self.learner_model.bounds)

        self.grid_size = grid_size

        self.possible_design = possible_design

        self.post = None

        self.grid_param = np.asarray(list(
            product(range(self.grid_size), repeat=self.n_param)))

        print(self.grid_param.shape)

        self.hist = []

        self.log_lik = np.zeros((len(possible_design), len(self.grid_param)))

    # def p_theta(self):
    #     return np.random.random(self.space)

    def reset(self):
        self.hist = []

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
                self.log_lik[i, j] = np.log(self._p_obs(x, param))
        # dim_p_obs = len(self.p_obs.shape)
        # y = self.y_obs.reshape(make_vector_shape(dim_p_obs + 1, dim_p_obs))
        # p = np.expand_dims(self.p_obs, dim_p_obs)

        # return  #log_lik_bernoulli(y, p)

    def _update_mutual_info(self):

        # self.log_post += self.log_lik[idx_design, :, idx_response].flatten()
        # self.log_post -= logsumexp(self.log_post)

        self._compute_log_lik()
        ll = self.log_lik

        lp = np.ones(self.grid_param.shape[1])
        self.log_prior = lp - logsumexp(lp)
        self.log_post = self.log_prior.copy()

        # Calculate the marginal log likelihood.
        lp = self.log_post
        # lp = expand_multiple_dims(self.log_post, 1, 1)
        mll = logsumexp(self.log_lik + lp, axis=1)
        self.marg_log_lik = mll  # shape (num_design, num_response)

        post = np.exp(self.log_post)

        # Calculate the marginal entropy and conditional entropy.
        self.ent_obs = -np.multiply(np.exp(ll), ll).sum(-1)
        self.ent_marg = -np.sum(np.exp(mll) * mll, -1)  # shape (num_designs,)
        self.ent_cond = np.sum(
            post * self.ent_obs, axis=1)  # shape (num_designs,)

        # Calculate the mutual information.
        self.mutual_info = self.ent_marg - \
                           self.ent_cond  # shape (num_designs,)

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
        elif kind == 'random':
            idx_design = np.random.random(self.n_design)
        else:
            raise ValueError(
                'The argument kind should be "optimal" or "random".')
        return idx_design

    def update(self, idx_design):
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
            Any kinds of observed response
        """
        # if not isinstance(design, pd.Series):
        #     design = pd.Series(design, index=self.task.designs)
        #
        # idx_design = get_nearest_grid_index(design, self.grid_design)
        # idx_response = get_nearest_grid_index(
        #     pd.Series(response), self.grid_response)

        self.hist.append(idx_design)

        self._update_mutual_info()

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
        "b0": (-5, 10),
        "b1": (-5, 10)
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

        logit = self.b0 + item * self.b1
        p_obs = 1. / (1 + np.exp(-logit))

        return p_obs

    def learn(self, **kwargs):
        pass


def main():

    print("Preparing...")

    np.random.seed(123)

    possible_design = np.arange(10)

    # student_model = HalfLife
    # true_param = {
    #     "beta": 0.02,
    #     "alpha": 0.2
    # }
    learner_model = FakeModel
    true_param = {
        "b0": 5,
        "b1": 2,
    }

    param = sorted(learner_model.bounds.keys())

    learner = learner_model(param=true_param)
    engine = Adaptive(learner_model=learner_model,
                      possible_design=possible_design)

    print("Ready to compute")

    design_types = ['optimal', 'random']

    num_trial = 100  # number of trials

    post_means = {pr: {d: np.zeros(num_trial)
                       for d in design_types}
                  for pr in param}
    post_sds = {pr: {d: np.zeros(num_trial)
                     for d in design_types}
                for pr in param}

    # Run simulations for three designs
    for design_type in design_types:
        # Reset the engine as an initial state
        engine.reset()

        for trial in tqdm(range(num_trial)):

            print(f"Computing results for design '{design_type}'")

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
