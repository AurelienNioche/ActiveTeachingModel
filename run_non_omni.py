import numpy as np
from scipy.special import logsumexp
import sys
from tqdm import tqdm
import itertools
import datetime

from exploration_exploitation import backward_induction


def calculate_prob(alpha, beta, delta, n_pres):
    """A function to compute the probability of a positive response."""
    p_obs = np.exp(-delta*alpha*(n_pres-1)*beta)
    return p_obs


def run_non_omni(n_item, param, review_ts, eval_ts, thr):
    np.random.seed(123)

    n_design = 1000
    bounds_design = 1, n_design
    design = np.linspace(*bounds_design, n_design)

    # (2e-07, 0.025)
    bounds = ((2e-07, 0.025), (0.0001, 0.9999))

    n_grid_param = 10
    methods = np.geomspace, np.linspace
    grid = np.array(list(itertools.product(
        *(m(*b, n_grid_param) for m, b in zip(methods, bounds)))))

    n_param_set, n_param = grid.shape

    print("n param set", n_param_set)

    alpha, beta = param

    lp = np.ones(n_param_set)
    lp -= logsumexp(lp)

    # Expected value
    ep = np.dot(np.exp(lp), grid)

    n_pres = np.zeros(n_item, dtype=int)
    last_pres = np.zeros(n_item)

    for i, ts in enumerate(review_ts):

        # post = np.exp(lp)

        # # # Compute likelihood
        # p_one = np.array([calculate_prob(delta=d,
        #                                  alpha=grid[:, 0],
        #                                  beta=grid[:, 1],
        #                                  n_pres=t)
        #                   for d in design])  # shape: (n design, n param set)
        #
        # p_obs = np.exp(-last_pres * alpha * (n_pres - 1) * beta)
        # p_zero = 1 - p_one
        # p = np.zeros((n_design, n_param_set, 2))
        # p[:, :, 0] = p_zero
        # p[:, :, 1] = p_one
        # log_lik = np.log(p + np.finfo(np.float).eps)
        #
        # # Compute entropy of LL
        # ent_obs = -np.multiply(np.exp(log_lik), log_lik).sum(-1)
        #
        # # Calculate the marginal log likelihood
        # # shape (num_design, num_response)
        # extended_lp = np.expand_dims(np.expand_dims(lp, 0), -1)
        # mll = logsumexp(log_lik + extended_lp, axis=1)
        #
        # # Calculate the marginal entropy and conditional entropy.
        # ent_marg = -np.sum(np.exp(mll) * mll, -1)  # shape (num_designs,)
        # ent_cond = np.sum(post * ent_obs, axis=1)  # shape (num_designs,)
        #
        # # Calculate the mutual information.
        # mutual_info = ent_marg - ent_cond  # shape (num_designs,)
        #
        # # Select design
        # d_idx = np.argmax(mutual_info)
        # d = design[d_idx]

        if np.max(n_pres) == 0:
            item = 0
        else:
            u = backward_induction.run(
                review_ts=review_ts,
                param=ep,
                thr=thr,
                eval_ts=eval_ts,
                n_pres_current=n_pres,
                idx_ts_current=i)

            item = np.argmax(u)

        # Produce response
        p_r = np.exp(-last_pres[item] * alpha * (n_pres[item] - 1) * beta)
        resp = p_r > np.random.random()

        if n_pres[item] > 0:
            # Get likelihood response
            log_lik_r = np.exp(-last_pres[item] * grid[:, 0] * (n_pres[item] - 1) * grid[:, 1])
            if not resp:
                log_lik_r = 1 - log_lik_r

            # Update prior
            lp += log_lik_r
            lp -= logsumexp(lp)

            # Expected value
            ep = np.dot(np.exp(lp), grid)

        # # Sd
        # delta = grid - ep
        # post_cov = np.dot(delta.T, delta * np.exp(lp).reshape(-1, 1))
        # sdp = np.sqrt(np.diag(post_cov))
        n_pres[item] += 1
        last_pres[item] = ts

    n_pres = np.zeros(n_item, dtype=int)
    last_pres = np.zeros(n_item)

    for i, ts in enumerate(review_ts):

        if np.max(n_pres) == 0:
            item = 0
        else:
            u = backward_induction.run(
                review_ts=review_ts,
                param=param,
                thr=thr,
                eval_ts=eval_ts,
                n_pres_current=n_pres,
                idx_ts_current=i)

            item = np.argmax(u)

        n_pres[item] += 1
        last_pres[item] = ts

    seen = n_pres > 0

    init_forget, rep_effect = param

    log_p_seen = \
        -init_forget \
        * (1 - rep_effect) ** (n_pres[seen] - 1) \
        * (eval_ts - last_pres[seen])

    n_learnt = np.sum(log_p_seen > np.log(thr))
    print("n learnt", n_learnt)
    print()


def main():

    np.random.seed(123)

    n_item = 100

    param = [0.0006, 0.44]
    param = np.asarray(param)

    ss_n_iter = 100
    time_per_iter = 4
    n_sec_day = 24 * 60 ** 2
    n_ss = 6
    eval_ts = n_ss * n_sec_day
    review_ts = np.hstack([
            np.arange(x, x + (ss_n_iter * time_per_iter), time_per_iter)
            for x in np.arange(0, n_sec_day * n_ss, n_sec_day)
        ])

    thr = 0.90

    a = datetime.datetime.now()

    run_non_omni(
        n_item=n_item, param=param, review_ts=review_ts,
        eval_ts=eval_ts, thr=thr)

    print("[Time to execute ", datetime.datetime.now() - a, "]")


if __name__ == "__main__":
    main()