# %%
import os
import numpy as np

from scipy.special import logsumexp
from tqdm import tqdm

from utils.decorator import use_pickle

from adaptive_teaching.constants import \
    POST_MEAN, POST_SD, \
    P, P_SEEN, FR_SEEN, N_SEEN, HIST, SUCCESS, TIMESTAMP, OBJECTIVE

from adaptive_teaching.teacher.leitner import Leitner


from adaptive_teaching.simplified.compute import compute_grid_param, \
    post_mean, post_sd

from adaptive_teaching.simplified import psychologist
from adaptive_teaching.simplified import teacher



EPS = np.finfo(np.float).eps
FIG_FOLDER = os.path.join("fig", "scenario")

LEITNER = "Leitner"
PSYCHOLOGIST = "Psychologist"
ADAPTIVE = "Adaptive"
TEACHER = "Teacher"
TEACHER_OMNISCIENT = "TeacherOmniscient"


@use_pickle
def run_n_days(
        learner,
        n_day, n_item, grid_size, param, seed,
        condition, n_iter_session=150, sec_per_iter=2,
        bounds=None,
        param_labels=None,
        using_multiprocessing=False,
):

    if bounds is None:
        bounds = learner.bounds

    if param_labels is None:
        param_labels = learner.param_labels

    n_iter_break = int((60**2 * 24) / sec_per_iter - n_iter_session)
    n_iteration = n_iter_session*n_day

    post_means = {pr: np.zeros(n_iteration) for pr in param_labels}
    post_sds = {pr: np.zeros(n_iteration) for pr in param_labels}

    p = np.zeros((n_item, n_iteration))

    hist = np.zeros(n_iteration, dtype=int)
    success = np.zeros(n_iteration, dtype=bool)
    timestamp = np.zeros(n_iteration)

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

    if condition == LEITNER:
        leitner = Leitner(task_param={'n_item': n_item})
    else:
        leitner = None

    pm, ps = None, None

    np.random.seed(seed)

    c_iter_session = 0
    t = 0

    if using_multiprocessing:
        iterator = range(n_iteration)
    else:
        iterator = tqdm(range(n_iteration))

    for it in iterator:

        log_lik = learner.log_lik(grid_param=grid_param,
                                  delta=delta,
                                  n_pres=n_pres,
                                  n_success=n_success,
                                  hist=hist,
                                  timestamps=timestamp,
                                  t=t)

        if condition == PSYCHOLOGIST:

            i = psychologist.get_item(
                log_post=log_post,
                log_lik=log_lik,
                n_item=n_item,
                grid_param=grid_param,
                delta=delta,
                n_pres=n_pres,
                n_success=n_success,
                hist=hist,
                learner=learner
            )

        elif condition == TEACHER:

            if t == 0:
                i = np.random.randint(n_item)
            else:
                i = teacher.get_item(
                    n_pres=n_pres,
                    n_success=n_success,
                    param=pm,
                    delta=delta,
                    hist=hist,
                    learner=learner,
                    timestamps=timestamp,
                    t=t
                )

        elif condition == TEACHER_OMNISCIENT:

            i = teacher.get_item(
                n_pres=n_pres,
                n_success=n_success,
                param=param,
                delta=delta,
                hist=hist,
                learner=learner,
                timestamps=timestamp,
                t=t
            )

        elif condition == ADAPTIVE:

            if t == 0:
                i = np.random.randint(n_item)

            elif np.all([ps[i] < 0.10 *
                         (bounds[i][1] - bounds[i][0])
                         for i in range(len(bounds))]):
                i = teacher.get_item(
                    n_pres=n_pres,
                    n_success=n_success,
                    param=param,
                    delta=delta,
                    hist=hist,
                    learner=learner,
                    timestamps=timestamp,
                    t=t
                )

            else:
                i = psychologist.get_item(
                    log_post=log_post,
                    log_lik=log_lik,
                    n_item=n_item,
                    grid_param=grid_param,
                    delta=delta,
                    n_pres=n_pres,
                    n_success=n_success,
                    hist=hist,
                    learner=learner,
                )

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
            timestamps=timestamp,
            t=t
        )

        response = p_recall > np.random.random()

        if condition == LEITNER:
            leitner.update(item=i, response=response)

        # Update prior
        log_post += log_lik[i, :, int(response)].flatten()
        log_post -= logsumexp(log_post)

        timestamp[it] = t
        hist[it] = i

        # Make the user learn
        # Increment delta for all items
        delta[:] += 1
        # ...except the one for the selected design that equal one
        delta[i] = 1
        n_pres[i] += 1
        n_success[i] += int(response)

        t += 1

        # Compute post mean and std
        pm = post_mean(grid_param=grid_param, log_post=log_post)
        ps = post_sd(grid_param=grid_param, log_post=log_post)

        # Backup the mean/std of post dist
        for i, pr in enumerate(param_labels):
            post_means[pr][it] = pm[i]
            post_sds[pr][it] = ps[i]

        # Backup prob recall / forgetting rates
        if hasattr(learner, 'fr_p_seen'):
            fr_seen_t, p_seen_t = \
                learner.fr_p_seen(
                    n_pres=n_pres,
                    n_success=n_success,
                    param=param,
                    delta=delta)
        else:
            p_seen_t = learner.p_seen(
                n_pres=n_pres,
                n_success=n_success,
                param=param,
                delta=delta,
                hist=hist,
                timestamps=timestamp,
                t=t
            )

            fr_seen_t = []

        # Backup
        seen = n_pres[:] > 0
        fr_seen.append(fr_seen_t)
        p_seen.append(p_seen_t)
        p[seen, it] = p_seen_t
        n_seen[it] = np.sum(seen)
        success[it] = int(response)

        c_iter_session += 1
        if c_iter_session >= n_iter_session:
            delta[:] += n_iter_break
            t += n_iter_break
            c_iter_session = 0

    results = {
        N_SEEN: n_seen,
        P: p,
        P_SEEN: p_seen,
        FR_SEEN: fr_seen,
        POST_MEAN: post_means,
        POST_SD: post_sds,
        HIST: hist,
        SUCCESS: success,
        TIMESTAMP: timestamp
    }

    return results


