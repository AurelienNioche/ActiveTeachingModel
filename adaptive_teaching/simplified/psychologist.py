import numpy as np

from scipy.special import logsumexp

from adaptive_teaching.simplified.learner import log_p_grid

# def compute_expected_gain(grid_param, log_lik, log_post, bounds):
#
#     n_param = grid_param.shape[1]
#     gain = np.zeros(n_param)
#     for response in (0, 1):
#         ll_item_response = log_lik[:, int(response)].flatten()
#
#         new_log_post = log_post + ll_item_response
#         new_log_post -= logsumexp(new_log_post)
#         gain_resp = - post_sd(grid_param=grid_param,
#                               log_post=new_log_post)
#
#         p = np.sum(np.exp(ll_item_response + log_post))
#
#         for i in range(n_param):
#             gain[i] += p * (gain_resp[i] / (bounds[i][1] - bounds[i][0]))
#
#     # for i in range(n_param):
#     #     _min = np.min(gain[i])
#     #     _max = np.max(gain[i])
#     #     if _max - _min > 0:
#     #         gain[i] = (gain[i] - _min) / (_max - _min)
#     #     else:
#     #         gain[i] = 0.5
#
#     return np.mean(gain)


def compute_expected_gain(log_lik, log_post):

    gain = 0
    for response in (0, 1):
        ll_item_response = log_lik[:, int(response)].flatten()

        new_log_post = log_post + ll_item_response
        new_log_post -= logsumexp(new_log_post)
        gain_resp = np.std(np.exp(new_log_post))

        p = np.sum(np.exp(ll_item_response + log_post))

        gain += p * gain_resp

    return np.mean(gain)


def get_item(log_post,
             log_lik,
             n_item,
             grid_param, delta, n_pres, n_success):

    items = np.arange(n_item)
    seen = n_pres[:] > 0

    n_seen = int(np.sum(seen))
    n_not_seen = n_item - n_seen

    not_seen = np.logical_not(seen)

    if n_seen == 0:
        return np.random.randint(n_item)

    elif n_not_seen == 0:
        item_sample = items[seen]

    else:
        item_sample = np.hstack((items[seen], [items[not_seen][0], ]))

    n_sample = len(item_sample)

    # u_t = np.zeros(n_sample)

    u_t_plus_one_if_pres = np.zeros(n_sample)

    u_t_plus_one_if_skipped = np.zeros(n_sample)

    for i, item in enumerate(item_sample):

        # u_t[i] = compute_expected_gain(
        #     grid_param=grid_param,
        #     log_post=log_post,
        #     log_lik=log_lik[i],
        #     bounds=bounds
        # )

        for response in (0, 1):

            p_resp = \
                np.sum(np.exp(log_lik[item, :, response].flatten() + log_post))

            ll_pres_i = log_p_grid(
                grid_param=grid_param,
                n_pres_i=n_pres[item] + 1,
                n_success_i=n_success[item] + response,
                delta_i=1,
                i=item
            )

            u_t_plus_one_if_pres_and_resp = compute_expected_gain(
                log_post=log_post,
                log_lik=ll_pres_i)

            u_t_plus_one_if_pres[i] += p_resp * u_t_plus_one_if_pres_and_resp

        ll_not_pres_i = log_p_grid(
            grid_param=grid_param,
            n_pres_i=n_pres[item],
            n_success_i=n_success[item],
            delta_i=delta[item] + 1,
            i=item
        )

        u_t_plus_one_if_skipped[i] = compute_expected_gain(
            log_post=log_post,
            log_lik=ll_not_pres_i)

    u = np.zeros(len(item_sample))

    for i in range(n_sample):
        max_u_skipped = np.max(
            u_t_plus_one_if_skipped[np.arange(n_sample) != i])
        # print(f"{i}/{n_sample-1}: u skipped {max_u_skipped}")
        u_presented = u_t_plus_one_if_pres[i]
        # print(f"{i}/{n_sample-1}: u present {u_presented}")
        u[i] = max(u_presented, max_u_skipped)

    return np.random.choice(
        item_sample[u[:] == np.max(u)])
