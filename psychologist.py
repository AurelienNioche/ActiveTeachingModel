# %%
import os
import numpy as np
from itertools import product
from scipy.special import logsumexp
from tqdm import tqdm

EPS = np.finfo(np.float).eps

from adaptive_teaching.constants import \
    POST_MEAN, POST_SD, \
    P, P_SEEN, FR_SEEN, N_SEEN

FIG_FOLDER = os.path.join("fig", "adaptive")

from adaptive_teaching.plot import \
    fig_parameter_recovery,  fig_p_recall_item

from adaptive_teaching.teacher.leitner import Leitner


LEITNER = "Leitner"
PSYCHOLOGIST = "Psychologist"


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
    return p


def learner_p(item, param, n_pres_minus_one, delta, seen):
    alpha, beta = param
    _seen = seen[item] == 1
    if _seen:
        p = np.exp(
            -alpha
            * (1 - beta) ** n_pres_minus_one[item]
            * delta[item])
    else:
        p = 0

    return p


def learner_fr_p_seen(seen, param, n_pres_minus_one, delta):
    alpha, beta = param
    fr = alpha * (1 - beta) ** n_pres_minus_one[seen]

    p = np.exp(-fr * delta[seen])
    return fr, p


# %%

def compute_grid_param(grid_size, bounds):
    return np.asarray(list(
        product(*[
            np.linspace(*bounds[key], grid_size)
            for key in sorted(bounds)])
    ))


# %%

def compute_log_lik(grid_param, seen, delta, n_pres_minus_one,
                    items):
    n_item = len(items)
    n_param_set = len(grid_param)
    log_lik = np.zeros((n_item, n_param_set, 2))
    for i, item in enumerate(items):
        log_lik[i, :, :] = np.log(
            p_grid(grid_param=grid_param,
                   i=i, seen=seen, delta=delta,
                   n_pres_minus_one=n_pres_minus_one)
            + EPS)

    return log_lik


# %%

def mutual_info_one_time_step(ll, lp):
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

def update_learner(delta, seen=None, n_pres_minus_one=None, item=None):
    # Increment delta for all items
    new_delta = delta + 1
    new_seen = seen.copy()
    new_n_pres_minus_one = n_pres_minus_one.copy()

    # ...except the one for the selected design that equal one
    if item is not None:
        new_delta[item] = 1
        new_seen[item] = 1
        new_n_pres_minus_one[item] += 1

    return new_delta, new_seen, new_n_pres_minus_one


def compute_mutual_info(log_post, n_item, seen,
                        grid_param, delta, n_pres_minus_one):
    n_param_set = len(grid_param)

    items = np.arange(n_item)

    n_seen = int(np.sum(seen))
    n_not_seen = n_item - n_seen

    not_seen = np.logical_not(seen)

    if n_seen == 0:
        return np.zeros(n_item)

    n_sample = min(n_seen + 1, n_item)

    items_seen = items[seen]

    if n_not_seen > 0:
        item_not_seen = items[not_seen][0]
        item_sample = list(items_seen) + [item_not_seen, ]
    else:
        item_sample = items_seen

    log_lik_sample = compute_log_lik(
        grid_param=grid_param,
        seen=seen, delta=delta,
        n_pres_minus_one=n_pres_minus_one,
        items=item_sample)

    log_lik = np.zeros((n_item, n_param_set, 2))

    log_lik[seen] = log_lik_sample[:n_seen]
    if n_not_seen:
        log_lik[not_seen] = log_lik_sample[-1]

    mi = mutual_info_one_time_step(log_lik, log_post)

    ll_after_pres = np.zeros((n_sample, n_param_set, 2))

    for i, item in enumerate(item_sample):
        # for response in (0, 1):
        new_delta, new_seen, new_n_pres_minus_one = \
            update_learner(delta=delta, seen=seen,
                           n_pres_minus_one=n_pres_minus_one,
                           item=item)

        ll_after_pres[i] += np.log(
            p_grid(grid_param, i,
                   new_seen, new_delta, new_n_pres_minus_one)
            + EPS)

    ll_without_pres = np.zeros((n_sample, n_param_set, 2))

    # Go to the future
    new_delta, new_seen, new_n_pres_minus_one = \
        update_learner(delta=delta, seen=seen,
                       n_pres_minus_one=n_pres_minus_one,
                       item=None)

    for i, item in enumerate(item_sample):
        ll_without_pres[i] = np.log(
            p_grid(grid_param, i,
                   new_seen, new_delta, new_n_pres_minus_one)
            + EPS)

    ll_after_pres_full = np.zeros(log_lik.shape)
    ll_without_pres_full = np.zeros(log_lik.shape)

    ll_after_pres_full[seen] = ll_after_pres[:n_seen]

    ll_without_pres_full[seen] = ll_without_pres[:n_seen]

    if n_not_seen:
        ll_after_pres_full[not_seen] = ll_after_pres[-1]
        ll_without_pres_full[not_seen] = ll_without_pres[-1]

    max_info_next_time_step = np.zeros(n_item)

    max_info_next_time_step[:] = np.zeros(n_item)
    for i in range(n_item):
        max_info_next_time_step[i] = \
            compute_max_info_time_step(
                n_item, n_param_set, i,
                ll_without_pres_full,
                ll_after_pres_full,
                log_post)

    mi[:] = mi + max_info_next_time_step
    return mi


def compute_max_info_time_step(n_item, n_param_set, i,
                               ll_without_pres_full,
                               ll_after_pres_full,
                               log_post
                               ):
    ll_t_plus_one = np.zeros((n_item, n_param_set, 2))

    ll_t_plus_one[:] = ll_without_pres_full[:]
    ll_t_plus_one[i] = ll_after_pres_full[i]

    mutual_info_t_plus_one_given_i = \
        mutual_info_one_time_step(ll_t_plus_one,
                                  log_post)

    return np.max(mutual_info_t_plus_one_given_i)


def get_item(log_post, n_item, seen,
             grid_param, delta, n_pres_minus_one):
    mi = compute_mutual_info(log_post, n_item, seen,
                             grid_param, delta, n_pres_minus_one)

    return np.random.choice(
        np.arange(n_item)[mi[:] == np.max(mi)]
    )


# %%

def post_mean(log_post, grid_param) -> np.ndarray:
    """
    A vector of estimated means for the posterior distribution.
    Its length is ``n_param_set``.
    """
    return np.dot(np.exp(log_post), grid_param)


def post_cov(grid_param, log_post) -> np.ndarray:
    """
    An estimated covariance matrix for the posterior distribution.
    Its shape is ``(num_grids, n_param_set)``.
    """
    # shape: (N_grids, N_param)
    d = grid_param - post_mean(log_post, grid_param)
    return np.dot(d.T, d * np.exp(log_post).reshape(-1, 1))


def post_sd(grid_param, log_post) -> np.ndarray:
    """
    A vector of estimated standard deviations for the posterior
    distribution. Its length is ``n_param_set``.
    """
    _post_cov = post_cov(grid_param, log_post)
    return np.sqrt(np.diag(_post_cov))


# def update(log_post, log_lik, item, response):
#     r"""
#     Update the posterior :math:`p(\theta | y_\text{obs}(t), d^*)` for
#     all discretized values of :math:`\theta`.
#
#     .. math::
#         p(\theta | y_\text{obs}(t), d^*) =
#             \frac{ p( y_\text{obs}(t) | \theta, d^*) p_t(\theta) }
#                 { p( y_\text{obs}(t) | d^* ) }
#     """
#
#     log_post += log_lik[item, :, int(response)].flatten()
#     log_post -= logsumexp(log_post)
#
#     return log_post


# %%

def run(n_trial, n_item, bounds, grid_size, param_labels, param, seed,
        condition):

    post_means = {pr: np.zeros(n_trial) for pr in param_labels}
    post_sds = {pr: np.zeros(n_trial) for pr in param_labels}

    p = np.zeros((n_item, n_trial))

    p_seen = []
    fr_seen = []

    hist_item = np.zeros(n_trial, dtype=int)
    hist_success = np.zeros(n_trial, dtype=bool)

    n_seen = np.zeros(n_trial, dtype=int)

    grid_param = compute_grid_param(bounds=bounds, grid_size=grid_size)
    n_param_set = len(grid_param)
    lp = np.ones(n_param_set)
    log_post = lp - logsumexp(lp)

    # Create learner and engine

    np.random.seed(seed)

    delta = np.zeros(n_item, dtype=int)
    n_pres_minus_one = np.zeros(n_item, dtype=int)
    seen = np.zeros(n_item, dtype=bool)

    if condition == LEITNER:
        leitner_teacher = Leitner(task_param={'n_item': n_item})
    else:
        leitner_teacher = None

    for t in tqdm(range(n_trial)):

        log_lik = compute_log_lik(grid_param=grid_param, seen=seen,
                                  delta=delta,
                                  n_pres_minus_one=n_pres_minus_one,
                                  items=np.arange(n_item))

        if condition == PSYCHOLOGIST:

            item = get_item(log_post, n_item, seen,
                            grid_param, delta, n_pres_minus_one)
        else:
            item = leitner_teacher.ask()

        p_recall = learner_p(item=item,
                             param=param,
                             delta=delta,
                             n_pres_minus_one=n_pres_minus_one,
                             seen=seen)

        response = p_recall > np.random.random()

        if condition == LEITNER:
            leitner_teacher.update(item=item, response=response)

        # Update prior
        log_post += log_lik[item, :, int(response)].flatten()
        log_post -= logsumexp(log_post)

        # Make the user learn
        # Increment delta for all items
        delta += 1
        # ...except the one for the selected design that equal one
        delta[item] = 1
        seen[item] = 1
        if t > 0:
            n_pres_minus_one[item] += 1

        # Compute post mean and std
        pm = post_mean(grid_param=grid_param, log_post=log_post)
        ps = post_sd(grid_param=grid_param, log_post=log_post)

        # Backup the mean/std of post dist
        for i, pr in enumerate(param_labels):
            post_means[pr][t] = pm[i]
            post_sds[pr][t] = ps[i]

        # Backup prob recall / forgetting rates
        fr_seen_t, p_seen_t = \
            learner_fr_p_seen(seen=seen, param=param,
                              n_pres_minus_one=n_pres_minus_one,
                              delta=delta)

        p[seen, t] = p_seen_t

        fr_seen.append(fr_seen_t)
        p_seen.append(p_seen_t)

        # Backup history
        n_seen[t] = np.sum(seen)

        hist_item[t] = item
        hist_success[t] = response

    return {
        N_SEEN: n_seen,
        P: p,
        P_SEEN: p_seen,
        FR_SEEN: fr_seen,
        POST_MEAN: post_means,
        POST_SD: post_sds
    }


def main():

    seed = 0
    n_trial = 100
    n_item = 100

    grid_size = 20

    bounds = {
        'alpha': (0.00, 1.0),
        'beta': (0.00, 1.0),
    }

    param = 0.05, 0.2

    param_labels = sorted(bounds.keys())

    condition_labels = LEITNER, PSYCHOLOGIST

    results = {}
    for cd in condition_labels:
        results[cd] = run(
            condition=cd,
            n_item=n_item,
            n_trial=n_trial,
            bounds=bounds,
            grid_size=grid_size,
            param_labels=param_labels,
            param=param,
            seed=seed)

    data_type = (POST_MEAN, POST_SD, P, P_SEEN, FR_SEEN, N_SEEN)
    data = {dt: {} for dt in data_type}

    for cd in condition_labels:
        for dt in data_type:
            print(results[cd].keys())
            d = results[cd][dt]
            data[dt][cd] = d

    fig_parameter_recovery(param=param_labels,
                           condition_labels=condition_labels,
                           post_means=data[POST_MEAN],
                           post_sds=data[POST_SD],
                           true_param={param_labels[i]: param[i]
                                       for i in range(len(param))},
                           num_trial=n_trial,
                           fig_name=None,
                           fig_folder=None)

    fig_p_recall_item(
        p_recall=data[P], condition_labels=condition_labels,
        fig_name=None, fig_folder=None)


# %%

if __name__ == "__main__":
    main()
