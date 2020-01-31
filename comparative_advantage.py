# %%
import os
import numpy as np
from itertools import product
from scipy.special import logsumexp
from tqdm import tqdm

from model.simplified.compute import \
    compute_grid_param, \
    post_mean, post_sd
from utils.decorator import use_pickle

from model.constants import \
    POST_MEAN, POST_SD, \
    P, P_SEEN, FR_SEEN, N_SEEN, HIST, SUCCESS

from model.plot import \
    fig_parameter_recovery,  fig_p_recall_item, fig_p_recall, fig_n_seen

from model.teacher.leitner import Leitner

from model.simplified.learner import ExponentialForgetting, \
    ActR, ExponentialForgettingAsymmetric

from model.plot.comparison import phase_diagram


from model.simplified import psychologist
from model.simplified import teacher

from model.simplified.scenario import run_n_days

EPS = np.finfo(np.float).eps
FIG_FOLDER = os.path.join("fig", "scenario")

LEITNER = "Leitner"
PSYCHOLOGIST = "Psychologist"
ADAPTIVE = "Adaptive"
TEACHER = "Teacher"
TEACHER_OMNISCIENT = "TeacherOmniscient"


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


def objective(results,):
    p_seen = results[P_SEEN]
    return np.sum(p_seen[-1][:] > 0.80)

    # threshold_item = 30
    # threshold_time = 10
    #
    # p_seen = results[P_SEEN]
    # n_seen = results[N_SEEN]
    #
    # n_trial = len(p_seen)
    #
    # c = 0
    # for t in range(n_trial):
    #
    #     if n_seen[t] >= threshold_item:
    #
    #         learnt = np.sum(np.asarray(p_seen[t])[:] > 0.80)
    #         if learnt >= threshold_item:
    #             # print("learnt", learnt, "c", c)
    #             c += 1
    #             if c == threshold_time:
    #                 return t
    #
    #     else:
    #         c = 0
    #
    # return n_trial


@use_pickle
def grid_exploration_objective(
        objective_function,
        parameter_values,
        bounds, grid_size, **kwargs):

    # Create a grid for each parameter
    param_grid = np.asarray(list(
            product(*parameter_values)
        ))

    n_sets = len(param_grid)

    # Container for log-likelihood
    obj = np.zeros(n_sets)

    # Loop over each value of the parameter grid for both parameters
    # for i in range(n_sets):
    for i in tqdm(range(n_sets)):

        # print(f"Total progression: {i / n_sets * 100:.2f}%")

        # Select the parameter to use
        param_to_use = param_grid[i]

        # Call the objective function of the optimizer
        obj[i] = objective_function(run(
            bounds=bounds,
            grid_size=grid_size,
            param=param_to_use,
            **kwargs
        ))

        # print(param_to_use, obj[i])

    return obj


def main_comparative_advantage():

    seed = 1
    n_iteration = 1000
    n_item = 300

    grid_size = 20

    learner = ExponentialForgetting
    bounds = (0.001, 0.04), (0.2, 0.5),
    param_labels = "alpha", "beta",

    condition_labels = \
        TEACHER, LEITNER  # , ADAPTIVE

    obj_values = dict()

    n_param = len(bounds)

    parameter_values = np.atleast_2d([
                np.linspace(
                    *bounds[i],
                    grid_size) for i in range(n_param)
    ])

    for cd in condition_labels:

        obj_values[cd] = grid_exploration_objective(
            learner=learner,
            objective_function=objective,
            parameter_values=parameter_values,
            condition=cd,
            n_item=n_item,
            n_iteration=n_iteration,
            bounds=bounds,
            grid_size=grid_size,
            param_labels=param_labels,
            seed=seed,
        )

        data = obj_values[cd]

        phase_diagram(parameter_values=parameter_values,
                      param_names=param_labels,
                      data=data,
                      fig_folder=FIG_FOLDER,
                      fig_name=f'phase_diagram_{cd}.pdf')

    data = (obj_values[TEACHER] - obj_values[LEITNER]) / obj_values[LEITNER] * 100
    # data[data[:] < 0] = 0

    phase_diagram(parameter_values=parameter_values,
                  param_names=param_labels,
                  data=data,
                  fig_folder=FIG_FOLDER,
                  fig_name=f'phase_diagram_teacher_better.pdf')

    # data = obj_values[TEACHER]-obj_values[LEITNER]
    # data[data[:] < 0] = 0
    #
    # phase_diagram(parameter_values=parameter_values,
    #               param_names=param_labels,
    #               data=data,
    #               fig_folder=FIG_FOLDER,
    #               fig_name=f'phase_diagram_leitner_better.pdf')


if __name__ == "__main__":
    main_comparative_advantage()
