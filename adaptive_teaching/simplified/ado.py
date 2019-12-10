import numpy as np

from scipy.special import logsumexp

from adaptive_teaching.simplified.learner import log_p_grid


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


def get_item(log_post, log_lik, n_item, seen,
             grid_param, delta, n_pres_minus_one):

    u = compute_mutual_info(
        log_post=log_post,
        log_lik=log_lik,
        n_item=n_item,
        seen=seen,
        grid_param=grid_param,
        delta=delta,
        n_pres_minus_one=n_pres_minus_one)

    return np.random.choice(
        np.arange(n_item)[u[:] == np.max(u)])
