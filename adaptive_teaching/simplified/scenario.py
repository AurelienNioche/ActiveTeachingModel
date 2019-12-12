# %%
import os
import numpy as np

from scipy.special import logsumexp
from tqdm import tqdm

from utils.decorator import use_pickle

from adaptive_teaching.constants import \
    POST_MEAN, POST_SD, \
    P, P_SEEN, FR_SEEN, N_SEEN, HIST, SUCCESS

from adaptive_teaching.teacher.leitner import Leitner

from adaptive_teaching.simplified.learner import learner_p, \
    learner_fr_p_seen

from adaptive_teaching.simplified.compute import compute_grid_param, \
    compute_log_lik, post_mean, post_sd

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
        n_day, n_item, bounds, grid_size, param_labels, param, seed,
        condition, n_iter_session=150, sec_per_iter=2,):

    n_iter_break = int((60**2 * 24) / sec_per_iter - n_iter_session)
    n_trial = n_iter_session*n_day

    post_means = {pr: np.zeros(n_trial) for pr in param_labels}
    post_sds = {pr: np.zeros(n_trial) for pr in param_labels}

    p = np.zeros((n_item, n_trial))

    hist = np.zeros(n_trial, dtype=int)
    success = np.zeros(n_trial, dtype=bool)

    p_seen = []
    fr_seen = []

    n_seen = np.zeros(n_trial, dtype=int)

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

    for t in tqdm(range(n_trial)):

        log_lik = compute_log_lik(grid_param=grid_param,
                                  delta=delta,
                                  n_pres=n_pres,
                                  n_success=n_success)

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
                    delta=delta
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

        p_recall = learner_p(
            param=param,
            delta_i=delta[i],
            n_pres_i=n_pres[i],
            n_success_i=n_success[i],
            i=i
        )

        response = p_recall > np.random.random()

        if condition == LEITNER:
            leitner.update(item=i, response=response)

        # Update prior
        log_post += log_lik[i, :, int(response)].flatten()
        log_post -= logsumexp(log_post)

        # Make the user learn
        # Increment delta for all items
        delta[:] += 1
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
            learner_fr_p_seen(n_pres=n_pres,
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

        c_iter_session += 1
        if c_iter_session >= n_iter_session:
            delta[:] += n_iter_break

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


