# %%

import numpy as np
from itertools import product
from scipy.special import logsumexp

EPS = np.finfo(np.float).eps

from multiprocessing import Pool

from adaptive_teaching.plot import fig_parameter_recovery


# %%

def p_grid(grid_param, i, seen, delta, n_pres_minus_one):

    n_param_set = len(grid_param)
    p = np.zeros((n_param_set, 2))

    i_has_been_seen = seen[i] == 1
    if i_has_been_seen:
        fr = grid_param[:, 0] \
             * (1 - grid_param[:, 1]) ** n_pres_minus_one[i]
        assert np.all(fr >= 0), f"{fr[fr <= 0][0]}"
        p[:, 1] = np.exp(
            - fr
            * delta[i])

    p[:, 0] = 1 - p[:, 1]


# %%

def compute_grid_param(grid_size, bounds):
    return np.asarray(list(
        product(*[
            np.linspace(*bounds[key], grid_size)
            for key in sorted(bounds)])
    ))


# %%

def compute_log_lik(n_item, grid_param, seen, delta, n_pres_minus_one):

    n_param_set = len(grid_param)
    log_lik = np.zeros((n_item, n_param_set, 2))
    for i in range(n_item):
        log_lik[i, :, :] = np.log(
            p_grid(grid_param, i, seen, delta, n_pres_minus_one)
            + EPS)


# %%

def mutual_info(ll, lp):

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
    mi = ent_mrg - ent_cond

    return mi


# %%

def compute_mutual_info(n_item, seen, grid_param, delta, n_pres_minus_one):

    n_param_set = len(grid_param)

    items = np.arange(n_item)

    n_seen = int(np.sum(seen))
    n_not_seen = n_item - n_seen

    not_seen = np.logical_not(seen)

    if n_seen == 0:
        raise Exception

    n_sample = min(n_seen + 1, n_item)

    log_lik = np.zeros((n_sample, n_param_set, 2))

    items_seen = items[seen]

    if n_not_seen > 0:
        item_not_seen = items[not_seen][0]
        item_sample = list(items_seen) + [item_not_seen, ]
    else:
        item_sample = items_seen

    for i, item in enumerate(item_sample):
        log_lik[i, :, :] = compute_log_lik(
            grid_param=grid_param,
            i=i, seen=seen, delta=delta,
            n_pres_minus_one=n_pres_minus_one,
            n_item=n_item)

    log_lik[seen] = log_lik[:n_seen]
    if n_not_seen:
        log_lik[not_seen] = log_lik[-1]

    mutual_info = mutual_info(log_lik, log_post)

    ll_after_pres = np.zeros((n_sample, self.n_param_set, 2))

    for i, item in enumerate(item_sample):

        for response in (0, 1):
            self.learner.update(item=item, response=response)

            ll_after_pres[i] += self._log_p(item)

            # Unlearn item
            self.learner.cancel_update()

    ll_without_pres = np.zeros((n_sample, n_param_set, 2))

    # Go to the future
    self.learner.update(item=None, response=None)

    for i, item in enumerate(item_sample):
        ll_without_pres[i] = self._log_p(item)

    # Cancel
    self.learner.cancel_update()

    self.ll_after_pres_full[seen] = ll_after_pres[:n_seen]

    self.ll_without_pres_full[seen] = ll_without_pres[:n_seen]

    if n_not_seen:
        self.ll_after_pres_full[not_seen] = ll_after_pres[-1]
        self.ll_without_pres_full[not_seen] = ll_without_pres[-1]

    max_info_next_time_step = np.zeros(self.n_item)

    with Pool() as pool:
        max_info_next_time_step[:] = \
            pool.map(self.compute_max_info_time_step, self.items)

    self.mutual_info[:] = mutual_info + max_info_next_time_step


def compute_max_info_time_step(self, i):
    ll_t_plus_one = np.zeros((self.n_item, self.n_param_set, 2))

    ll_t_plus_one[:] = self.ll_without_pres_full[:]
    ll_t_plus_one[i] = self.ll_after_pres_full[i]

    mutual_info_t_plus_one_given_i = \
        self._mutual_info(ll_t_plus_one,
                          self.log_post)

    return np.max(mutual_info_t_plus_one_given_i)


# %%

def post(log_post) -> np.ndarray:
    """Posterior distributions of joint parameter space"""
    return np.exp(log_post)


def post_mean(post, grid_param) -> np.ndarray:
    """
    A vector of estimated means for the posterior distribution.
    Its length is ``n_param_set``.
    """
    return np.dot(post, grid_param)


def post_cov(grid_param, post_mean, post) -> np.ndarray:
    """
    An estimated covariance matrix for the posterior distribution.
    Its shape is ``(num_grids, n_param_set)``.
    """
    # shape: (N_grids, N_param)
    d = grid_param - post_mean
    return np.dot(d.T, d * post.reshape(-1, 1))


def post_sd(post_cov) -> np.ndarray:
    """
    A vector of estimated standard deviations for the posterior
    distribution. Its length is ``n_param_set``.
    """
    return np.sqrt(np.diag(post_cov))


def update(log_post, log_lik, item, response):
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

    log_post += log_lik[item, :, int(response)].flatten()
    log_post -= logsumexp(log_post)

    return log_post


# %%

def run():
    n_trial = 100
    n_item = 10

    bounds = {
        'alpha': (0.00, 1.0),
        'beta': (0.00, 1.0),
    }

    param = sorted(bounds.keys())

    post_means = {pr: np.zeros(n_trial) for pr in param}
    post_sds = {pr: np.zeros(n_trial) for pr in param}

    p = np.zeros((n_item, n_trial))
    fr = np.zeros((n_item, n_trial))
    p_seen = []
    fr_seen = []

    hist_item = np.zeros(n_trial, dtype=int)
    hist_success = np.zeros(n_trial, dtype=bool)

    n_seen = np.zeros(n_trial, dtype=int)

    # Create learner and engine
    learner = learner_model(param=learner_param, task_param=task_param)
    engine = engine_model(
        teacher_model=teacher_model,
        teacher_param=teacher_param,
        learner_model=learner_model,
        task_param=task_param,
        **engine_param
    )

    np.random.seed(seed)

    for t in tqdm(range(n_trial)):

        # Compute an optimal design for the current trial
        item = engine.get_item()

        # Get a response using the optimal design
        p_recall = learner.p(item=item)

        response = p_recall > np.random.random()

        # Update the engine
        engine.update(item=item, response=response)

        # Make the user learn
        learner.update(item=item, response=response)

        # Backup the mean/std of post dist
        for i, pr in enumerate(param):
            post_means[pr][t] = engine.post_mean[i]
            post_sds[pr][t] = engine.post_sd[i]

        # Backup prob recall / forgetting rates
        fr[:, t], p[:, t] = \
            learner.forgetting_rate_and_p_all()

        fr_seen_t, p_seen_t = \
            learner.forgetting_rate_and_p_seen()

        fr_seen.append(fr_seen_t)
        p_seen.append(p_seen_t)

        # Backup history
        n_seen[t] = np.sum(learner.seen)

        hist_item[t] = item
        hist_success[t] = response

        fig_parameter_recovery(param=param, design_types=labels,
                               post_means=data[POST_MEAN],
                               post_sds=data[POST_SD],
                               true_param=learner_param,
                               num_trial=task_param['n_trial'],
                               fig_name=fig_name,
                               fig_folder=FIG_FOLDER)


run()


# %%
