# %%
import os
import numpy as np
from itertools import product
from scipy.special import logsumexp
from tqdm import tqdm

from utils.decorator import use_pickle

from adaptive_teaching.constants import \
    POST_MEAN, POST_SD, \
    P, P_SEEN, FR_SEEN, N_SEEN

from adaptive_teaching.plot import \
    fig_parameter_recovery,  fig_p_recall_item, fig_p_recall, fig_n_seen

from adaptive_teaching.teacher.leitner import Leitner

from adaptive_teaching.simplified.learner import log_p_grid, learner_p, \
    learner_fr_p_seen


from adaptive_teaching.simplified import psychologist
from adaptive_teaching.simplified import teacher

EPS = np.finfo(np.float).eps
FIG_FOLDER = os.path.join("fig", "adaptive")

LEITNER = "Leitner"
PSYCHOLOGIST = "Psychologist"
ADAPTIVE = "Adaptive"
TEACHER = "Teacher"
TEACHER_OMNISCIENT = "TeacherOmniscient"


# %%

def compute_grid_param(grid_size, bounds):
    return np.asarray(list(
        product(*[
            np.linspace(*b, grid_size)
            for b in bounds])))


# %%

def compute_log_lik(grid_param, delta, n_pres, n_success):

    n_item = len(n_pres)
    n_param_set = len(grid_param)

    log_lik = np.zeros((n_item, n_param_set, 2))

    for i in range(n_item):
        log_lik[i, :, :] = log_p_grid(
            grid_param=grid_param,
            delta_i=delta[i],
            n_pres_i=n_pres[i],
            n_success_i=n_success[i],
            i=i
        )

    return log_lik


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


# %%

@use_pickle
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
    n_pres = np.zeros(n_item, dtype=int)
    n_success = np.zeros(n_item, dtype=int)

    if condition == LEITNER:
        leitner = Leitner(task_param={'n_item': n_item})
    else:
        leitner = None

    pm, ps = None, None

    np.random.seed(seed)
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
    n_trial = 1000
    n_item = 50

    grid_size = 20

    # bounds = [(0.00, 1.0), ] * (n_item + 1)
    # param = np.hstack((np.random.uniform(0, 0.5, n_item), [0.05, 0.2]))

    param = 0.05, 0.2,
    bounds = (0., 1.), (0., 1.), (0., 1.),
    param_labels = "alpha", "beta",

    # param = 0.05, 0, 0.2
    # bounds = (0., 1.), (0., 1.), (0., 1.), (0., 1.)
    # param_labels = "alpha", "beta", "gamma"

    condition_labels = \
        TEACHER, LEITNER, ADAPTIVE

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

    fig_p_recall(data=data[P_SEEN], labels=condition_labels,
                 fig_name=None, fig_folder=None)

    fig_p_recall(
        y_label="Forgetting rates",
        data=data[FR_SEEN], labels=condition_labels,
        fig_name=None, fig_folder=None)

    fig_n_seen(
        data=data[N_SEEN], design_types=condition_labels,
        fig_name=None, fig_folder=None)


# %%

if __name__ == "__main__":
    main()
