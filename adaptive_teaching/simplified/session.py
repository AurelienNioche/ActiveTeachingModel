import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm

from adaptive_teaching.constants import N_SEEN, P, P_SEEN, FR_SEEN, POST_MEAN, \
    POST_SD, HIST, SUCCESS
from adaptive_teaching.simplified import psychologist, teacher
from adaptive_teaching.simplified.compute import compute_grid_param, post_mean, \
    post_sd
from adaptive_teaching.teacher.leitner import Leitner
from . labels import LEITNER, PSYCHOLOGIST, TEACHER, TEACHER_OMNISCIENT, ADAPTIVE
from utils.decorator import use_pickle


@use_pickle
def run(n_iteration, n_item, bounds, grid_size, param_labels, param, seed,
        condition, learner):

    post_means = {pr: np.zeros(n_iteration) for pr in param_labels}
    post_sds = {pr: np.zeros(n_iteration) for pr in param_labels}

    p = np.zeros((n_item, n_iteration))

    p_seen = []
    fr_seen = []

    n_seen = np.zeros(n_iteration, dtype=int)

    grid_param = compute_grid_param(bounds=bounds, grid_size=grid_size)
    n_param_set = len(grid_param)
    lp = np.ones(n_param_set)
    log_post = lp - logsumexp(lp)

    delta = np.zeros(n_item, dtype=int)
    n_pres = np.zeros(n_item, dtype=int)
    n_success = np.zeros(n_item, dtype=int)

    timestamps = np.full(n_iteration, -1)
    hist = np.full(n_iteration, -1, dtype=int)
    success = np.full(n_iteration, -1, dtype=bool)

    if condition == LEITNER:
        leitner = Leitner(task_param={'n_item': n_item})
    else:
        leitner = None

    pm, ps = None, None

    np.random.seed(seed)
    for t in tqdm(range(n_iteration)):

        log_lik = learner.log_lik(
            grid_param=grid_param,
            delta=delta,
            n_pres=n_pres,
            n_success=n_success,
            hist=hist,
            timestamps=timestamps,
            t=t
        )

        if condition == PSYCHOLOGIST:

            i = psychologist.get_item(
                log_post=log_post,
                log_lik=log_lik,
                n_item=n_item,
                grid_param=grid_param,
                delta=delta,
                n_pres=n_pres,
                n_success=n_success)

        elif condition == TEACHER:

            if t == 0:
                i = np.random.randint(n_item)
            else:
                i = teacher.get_item(
                    n_pres=n_pres,
                    n_success=n_success,
                    param=pm,
                    delta=delta,
                    learner=learner,
                    hist=hist,
                    timestamps=timestamps,
                    t=t
                )

        elif condition == TEACHER_OMNISCIENT:

            i = teacher.get_item(
                n_pres=n_pres,
                n_success=n_success,
                param=param,
                delta=delta
            )

        elif condition == ADAPTIVE:

            if t == 0:
                i = np.random.randint(n_item)

            elif np.all([ps[i] < 0.10 * (bounds[i][1] - bounds[i][0])
                         for i in range(len(bounds))]):
                i = teacher.get_item(
                    n_pres=n_pres,
                    n_success=n_success,
                    param=param,
                    delta=delta)

            else:
                i = psychologist.get_item(
                    log_post=log_post,
                    log_lik=log_lik,
                    n_item=n_item,
                    grid_param=grid_param,
                    delta=delta,
                    n_pres=n_pres,
                    n_success=n_success)

        elif condition == LEITNER:
            i = leitner.ask()

        else:
            raise ValueError("Condition not recognized")

        p_recall = learner.p(
            param=param,
            delta_i=delta[i],
            n_pres_i=n_pres[i],
            n_success_i=n_success[i],
            i=i,
            hist=hist,
            timestamps=timestamps,
            t=t,
        )

        response = p_recall > np.random.random()

        if condition == LEITNER:
            leitner.update(item=i, response=response)

        # Update prior
        log_post += log_lik[i, :, int(response)].flatten()
        log_post -= logsumexp(log_post)

        # Make the user learn
        # Increment delta for all items
        delta += 1
        # ...except the one for the selected design that equal one
        delta[i] = 1
        n_pres[i] += 1
        n_success[i] += int(response)

        # Compute post mean and std
        pm = post_mean(grid_param=grid_param, log_post=log_post)
        ps = post_sd(grid_param=grid_param, log_post=log_post)

        # Backup the mean/std of post dist
        for i, pr in enumerate(param_labels):
            post_means[pr][t] = pm[i]
            post_sds[pr][t] = ps[i]

        # Backup prob recall / forgetting rates
        fr_seen_t, p_seen_t = \
            learner.fr_p_seen(n_pres=n_pres,
                              n_success=n_success,
                              param=param,
                              delta=delta)

        # Backup
        seen = n_pres[:] > 0
        fr_seen.append(fr_seen_t)
        p_seen.append(p_seen_t)
        p[seen, t] = p_seen_t
        n_seen[t] = np.sum(seen)
        success[t] = int(response)
        hist[t] = i
        timestamps[t] = t

    return {
        N_SEEN: n_seen,
        P: p,
        P_SEEN: p_seen,
        FR_SEEN: fr_seen,
        POST_MEAN: post_means,
        POST_SD: post_sds,
        HIST: hist,
        SUCCESS: success
    }