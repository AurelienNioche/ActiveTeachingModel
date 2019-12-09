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

def log_p_grid(grid_param, seen_i, delta_i, n_pres_minus_one_i):

    n_param_set = len(grid_param)
    p = np.zeros((n_param_set, 2))

    if seen_i:
        fr = grid_param[:, 0] * (1 - grid_param[:, 1]) ** n_pres_minus_one_i

        assert np.all(fr >= 0), f"{fr[fr <= 0][0]}"

        p[:, 1] = np.exp(- fr * delta_i)

    p[:, 0] = 1 - p[:, 1]
    return np.log(p + EPS)


def learner_p(param, n_pres_minus_one_i, delta_i, seen_i):

    if seen_i:
        fr = param[0] * (1 - param[1]) ** n_pres_minus_one_i
        p = np.exp(- fr * delta_i)
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

def compute_log_lik(grid_param, seen, delta, n_pres_minus_one):

    n_item = len(seen)
    n_param_set = len(grid_param)

    log_lik = np.zeros((n_item, n_param_set, 2))

    for i in range(n_item):
        log_lik[i, :, :] = log_p_grid(
            grid_param=grid_param,
            seen_i=seen[i], delta_i=delta[i],
            n_pres_minus_one_i=n_pres_minus_one[i])

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
    return ent_mrg - ent_cond


# %%

def compute_mutual_info(log_post, log_lik, n_item, seen,
                        grid_param, delta, n_pres_minus_one):

    n_param_set = len(grid_param)
    items = np.arange(n_item)

    n_seen = int(np.sum(seen))
    n_not_seen = n_item - n_seen

    not_seen = np.logical_not(seen)

    if n_seen == 0:
        return np.zeros(n_item)
        # item_sample = np.array([0])

    elif n_not_seen == 0:
        item_sample = items[seen]

    else:
        item_sample = np.hstack((items[seen], [items[not_seen][0], ]))

    n_sample = len(item_sample)

    mi_t = mutual_info_one_time_step(ll=log_lik, lp=log_post)

    ll_pres_spl = np.zeros((n_sample, n_param_set, 2))
    ll_non_pres_spl = np.zeros((n_sample, n_param_set, 2))

    for i, item in enumerate(item_sample):

        ll_pres_spl[i] = log_p_grid(
            grid_param=grid_param,
            seen_i=True,
            n_pres_minus_one_i=n_pres_minus_one[item] + 1,
            delta_i=1
        )

        ll_non_pres_spl[i] = log_p_grid(
            grid_param=grid_param,
            seen_i=seen[item],
            n_pres_minus_one_i=n_pres_minus_one[item],
            delta_i=delta[item] + 1
        )

    ll_pres_full = np.zeros(log_lik.shape)
    ll_non_pres_full = np.zeros(log_lik.shape)

    ll_pres_full[seen] = ll_pres_spl[:n_seen]
    ll_non_pres_full[seen] = ll_non_pres_spl[:n_seen]

    if n_not_seen > 0:
        ll_pres_full[not_seen] = ll_pres_spl[-1]
        ll_non_pres_full[not_seen] = ll_non_pres_spl[-1]

    max_info_next_time_step = np.zeros(n_item)
    for i in range(n_item):

        ll_t_plus_one = np.zeros((n_item, n_param_set, 2))

        ll_t_plus_one[:] = ll_non_pres_full[:]
        ll_t_plus_one[i] = ll_pres_full[i]

        mi_t_plus_one_given_i = \
            mutual_info_one_time_step(ll=ll_t_plus_one, lp=log_post)

        max_info_next_time_step[i] = np.max(mi_t_plus_one_given_i)

    mi = np.zeros(n_item)
    mi[:] = mi_t + max_info_next_time_step
    return mi


def compute_expected_gain(grid_param, log_lik, log_post,
                          post_standard_deviation):

    gain = 0
    for response in (0, 1):
        ll_item_response = log_lik[:, int(response)].flatten()

        new_log_post = log_post + ll_item_response
        new_log_post -= logsumexp(new_log_post)
        gain_resp = post_standard_deviation \
            - post_sd(grid_param=grid_param,
                      log_post=new_log_post)

        gain += np.sum(np.exp(ll_item_response + log_post) * gain_resp[1])

    return gain


def get_item(log_post, log_lik, n_item, seen,
             grid_param, delta, n_pres_minus_one):
    # mi = compute_mutual_info(
    #     log_post=log_post,
    #     log_lik=log_lik,
    #     n_item=n_item,
    #     seen=seen,
    #     grid_param=grid_param,
    #     delta=delta,
    #     n_pres_minus_one=n_pres_minus_one)

    items = np.arange(n_item)

    n_seen = int(np.sum(seen))
    n_not_seen = n_item - n_seen

    not_seen = np.logical_not(seen)

    if n_seen == 0:
        return np.random.randint(n_item)
        # item_sample = np.array([0])

    elif n_not_seen == 0:
        item_sample = items[seen]

    else:
        item_sample = np.hstack((items[seen], [items[not_seen][0], ]))

    n_sample = len(item_sample)

    u_t = np.zeros(n_sample)

    u_t_plus_one_if_presented = np.zeros(n_sample)

    u_t_plus_one_if_skipped = np.zeros(n_sample)

    post_standard_deviation = post_sd(grid_param=grid_param, log_post=log_post)

    for i, item in enumerate(item_sample):

        u_t[i] = compute_expected_gain(
            grid_param=grid_param,
            log_post=log_post,
            post_standard_deviation=post_standard_deviation,
            log_lik=log_lik[i]
        )

        ll_pres_i = log_p_grid(
            grid_param=grid_param,
            seen_i=True,
            n_pres_minus_one_i=n_pres_minus_one[item] + 1,
            delta_i=1
        )

        u_t_plus_one_if_presented[i] = compute_expected_gain(
            grid_param=grid_param,
            log_post=log_post,
            post_standard_deviation=post_standard_deviation,
            log_lik=ll_pres_i
        )

        ll_not_pres_i = log_p_grid(
            grid_param=grid_param,
            seen_i=seen[item],
            n_pres_minus_one_i=n_pres_minus_one[item],
            delta_i=delta[item] + 1
        )

        u_t_plus_one_if_skipped[i] = compute_expected_gain(
            grid_param=grid_param,
            log_post=log_post,
            post_standard_deviation=post_standard_deviation,
            log_lik=ll_not_pres_i
        )

    u = np.zeros(len(item_sample))

    for i in range(n_sample):
        max_u_skipped = np.max(u_t_plus_one_if_skipped[np.arange(n_sample)!=i])
        # print(f"{i}/{n_sample-1}: u skipped {max_u_skipped}")
        u_presented = u_t_plus_one_if_presented[i]
        # print(f"{i}/{n_sample-1}: u present {u_presented}")
        u[i] = u_t[i] + u_presented + max(u_presented, max_u_skipped)

    return np.random.choice(
        item_sample[u[:] == np.max(u)]
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
    _post_mean = post_mean(log_post=log_post, grid_param=grid_param)
    d = grid_param - _post_mean
    return np.dot(d.T, d * np.exp(log_post).reshape(-1, 1))


def post_sd(grid_param, log_post) -> np.ndarray:
    """
    A vector of estimated standard deviations for the posterior
    distribution. Its length is ``n_param_set``.
    """
    _post_cov = post_cov(grid_param=grid_param, log_post=log_post)
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

    n_seen = np.zeros(n_trial, dtype=int)

    grid_param = compute_grid_param(bounds=bounds, grid_size=grid_size)
    n_param_set = len(grid_param)
    lp = np.ones(n_param_set)
    log_post = lp - logsumexp(lp)

    delta = np.zeros(n_item, dtype=int)
    n_pres_minus_one = np.zeros(n_item, dtype=int)
    seen = np.zeros(n_item, dtype=bool)

    if condition == LEITNER:
        leitner_teacher = Leitner(task_param={'n_item': n_item})
    else:
        leitner_teacher = None

    np.random.seed(seed)
    for t in tqdm(range(n_trial)):

        log_lik = compute_log_lik(grid_param=grid_param, seen=seen,
                                  delta=delta,
                                  n_pres_minus_one=n_pres_minus_one)

        if condition == PSYCHOLOGIST:

            i = get_item(log_post=log_post,
                         log_lik=log_lik,
                         n_item=n_item, seen=seen,
                         grid_param=grid_param,
                         delta=delta,
                         n_pres_minus_one=n_pres_minus_one)
        else:
            i = leitner_teacher.ask()

        p_recall = learner_p(
            param=param,
            delta_i=delta[i],
            n_pres_minus_one_i=n_pres_minus_one[i],
            seen_i=seen[i])

        response = p_recall > np.random.random()

        if condition == LEITNER:
            leitner_teacher.update(item=i, response=response)

        # Update prior
        log_post += log_lik[i, :, int(response)].flatten()
        log_post -= logsumexp(log_post)

        # Make the user learn
        # Increment delta for all items
        delta += 1
        # ...except the one for the selected design that equal one
        delta[i] = 1
        seen[i] = 1
        if t > 0:
            n_pres_minus_one[i] += 1

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

        # Backup
        fr_seen.append(fr_seen_t)
        p_seen.append(p_seen_t)
        p[seen, t] = p_seen_t
        n_seen[t] = np.sum(seen)

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
    n_trial = 500
    n_item = 200

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
