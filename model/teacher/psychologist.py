import numpy as np

from scipy.special import logsumexp

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


class Psychologist:

    @classmethod
    def is_confident(cls, post_std, bounds, thr=0.10):

        return np.all([post_std[i] < thr * (bounds[i][1] - bounds[i][0])
                      for i in range(len(bounds))])

    @classmethod
    def compute_expected_gain(cls, log_lik, log_post):

        gain = 0
        for response in (0, 1):
            ll_item_response = log_lik[:, int(response)].flatten()

            new_log_post = log_post + ll_item_response
            new_log_post -= logsumexp(new_log_post)
            gain_resp = np.std(np.exp(new_log_post))

            p = np.sum(np.exp(ll_item_response + log_post))

            gain += p * gain_resp

        return np.mean(gain)

    @classmethod
    def get_item(
            cls,
            learner,
            log_post,
            log_lik,
            n_item,
            grid_param, delta, n_pres, n_success,
            hist):

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
                    np.sum(np.exp(log_lik[item, :, response].flatten()
                                  + log_post))

                ll_pres_i = learner.log_p_grid(
                    grid_param=grid_param,
                    n_pres_i=n_pres[item] + 1,
                    n_success_i=n_success[item] + response,
                    delta_i=1,
                    i=item,
                    hist=hist + [item, ]
                )

                u_t_plus_one_if_pres_and_resp = cls.compute_expected_gain(
                    log_post=log_post,
                    log_lik=ll_pres_i)

                u_t_plus_one_if_pres[i] += p_resp * u_t_plus_one_if_pres_and_resp

            ll_not_pres_i = learner.log_p_grid(
                grid_param=grid_param,
                n_pres_i=n_pres[item],
                n_success_i=n_success[item],
                delta_i=delta[item] + 1,
                i=item,
                hist=hist + [-1, ]
            )

            u_t_plus_one_if_skipped[i] = cls.compute_expected_gain(
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

#
# def mutual_info_one_time_step(ll, lp):
#
#     lp_reshaped = lp.reshape((1, len(lp), 1))
#
#     # ll => likelihood
#     # shape (n_item, n_param_set, num_responses, )
#
#     # Calculate the marginal log likelihood.
#     # shape (n_item, num_responses, )
#     mll = logsumexp(ll + lp_reshaped, axis=1)
#
#     # Calculate the marginal entropy and conditional entropy.
#     # shape (n_item,)
#     ent_mrg = - np.sum(np.exp(mll) * mll, -1)
#
#     # Compute entropy obs -------------------------
#
#     # shape (n_item, n_param_set, num_responses, )
#     # shape (n_item, n_param_set, )
#     ent_obs = - np.multiply(np.exp(ll), ll).sum(-1)
#
#     # Compute conditional entropy -----------------
#
#     # shape (n_item,)
#     ent_cond = np.sum(np.exp(lp) * ent_obs, axis=1)
#
#     # Calculate the mutual information. -----------
#     # shape (n_item,)
#     return ent_mrg - ent_cond
#
#
# def compute_mutual_info(log_post, log_lik, n_item, seen,
#                         grid_param, delta, n_pres_minus_one):
#
#     n_param_set = len(grid_param)
#     items = np.arange(n_item)
#
#     n_seen = int(np.sum(seen))
#     n_not_seen = n_item - n_seen
#
#     not_seen = np.logical_not(seen)
#
#     if n_seen == 0:
#         return np.zeros(n_item)
#         # item_sample = np.array([0])
#
#     elif n_not_seen == 0:
#         item_sample = items[seen]
#
#     else:
#         item_sample = np.hstack((items[seen], [items[not_seen][0], ]))
#
#     n_sample = len(item_sample)
#
#     mi_t = mutual_info_one_time_step(ll=log_lik, lp=log_post)
#
#     ll_pres_spl = np.zeros((n_sample, n_param_set, 2))
#     ll_non_pres_spl = np.zeros((n_sample, n_param_set, 2))
#
#     for i, item in enumerate(item_sample):
#
#         ll_pres_spl[i] = log_p_grid(
#             grid_param=grid_param,
#             seen_i=True,
#             n_pres_minus_one_i=n_pres_minus_one[item] + 1,
#             delta_i=1
#         )
#
#         ll_non_pres_spl[i] = log_p_grid(
#             grid_param=grid_param,
#             seen_i=seen[item],
#             n_pres_minus_one_i=n_pres_minus_one[item],
#             delta_i=delta[item] + 1
#         )
#
#     ll_pres_full = np.zeros(log_lik.shape)
#     ll_non_pres_full = np.zeros(log_lik.shape)
#
#     ll_pres_full[seen] = ll_pres_spl[:n_seen]
#     ll_non_pres_full[seen] = ll_non_pres_spl[:n_seen]
#
#     if n_not_seen > 0:
#         ll_pres_full[not_seen] = ll_pres_spl[-1]
#         ll_non_pres_full[not_seen] = ll_non_pres_spl[-1]
#
#     max_info_next_time_step = np.zeros(n_item)
#     for i in range(n_item):
#
#         ll_t_plus_one = np.zeros((n_item, n_param_set, 2))
#
#         ll_t_plus_one[:] = ll_non_pres_full[:]
#         ll_t_plus_one[i] = ll_pres_full[i]
#
#         mi_t_plus_one_given_i = \
#             mutual_info_one_time_step(ll=ll_t_plus_one, lp=log_post)
#
#         max_info_next_time_step[i] = np.max(mi_t_plus_one_given_i)
#
#     mi = np.zeros(n_item)
#     mi[:] = mi_t + max_info_next_time_step
#     return mi
#
#
# def get_item(log_post, log_lik, n_item, seen,
#              grid_param, delta, n_pres_minus_one):
#
#     u = compute_mutual_info(
#         log_post=log_post,
#         log_lik=log_lik,
#         n_item=n_item,
#         seen=seen,
#         grid_param=grid_param,
#         delta=delta,
#         n_pres_minus_one=n_pres_minus_one)
#
#     return np.random.choice(
#         np.arange(n_item)[u[:] == np.max(u)])
