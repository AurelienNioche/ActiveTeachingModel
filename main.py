# %%
import os
import numpy as np
from itertools import product
from tqdm import tqdm

from adaptive_teaching.simplified.session import run
from utils.decorator import use_pickle

from adaptive_teaching.constants import \
    P_SEEN

from adaptive_teaching.simplified.learner import ExponentialForgetting

from adaptive_teaching.plot.comparison import phase_diagram
from adaptive_teaching.simplified.scenario import run_n_days

EPS = np.finfo(np.float).eps
FIG_FOLDER = os.path.join("fig", "scenario")


from adaptive_teaching.simplified.labels \
    import LEITNER, PSYCHOLOGIST, ADAPTIVE, TEACHER, TEACHER_OMNISCIENT


def truncate_10(x):
    return int(x/10) * 10


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
                      fig_name=f'phase_diagram_{cd}.pdf',
                      levels=np.linspace(np.min(data), np.max(data), 10))

    data = \
        (obj_values[TEACHER] - obj_values[LEITNER]) / obj_values[LEITNER] * 100
    # data[data[:] < 0] = 0

    phase_diagram(parameter_values=parameter_values,
                  param_names=param_labels,
                  data=data,
                  fig_folder=FIG_FOLDER,
                  fig_name=f'phase_diagram_teacher_better.pdf',
                  levels=np.linspace(truncate_10(np.min(data)),
                                     np.max(data)+10, 10))

    # data = obj_values[TEACHER]-obj_values[LEITNER]
    # data[data[:] < 0] = 0
    #
    # phase_diagram(parameter_values=parameter_values,
    #               param_names=param_labels,
    #               data=data,
    #               fig_folder=FIG_FOLDER,
    #               fig_name=f'phase_diagram_leitner_better.pdf')


@use_pickle
def grid_exploration_objective_n_days(
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
        obj[i] = objective_function(run_n_days(
            bounds=bounds,
            grid_size=grid_size,
            param=param_to_use,
            **kwargs
        ))

        # print(param_to_use, obj[i])

    return obj


def main_comparative_advantage_n_days():

    seed = 1
    n_iter_session = 150
    n_day = 20
    n_item = 1000

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

        obj_values[cd] = grid_exploration_objective_n_days(
            learner=learner,
            objective_function=objective,
            parameter_values=parameter_values,
            condition=cd,
            n_item=n_item,
            n_iter_session=n_iter_session,
            n_day=n_day,
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
                      fig_name=f'phase_diagram_{cd}.pdf',
                      levels=np.linspace(np.min(data), np.max(data), 10))

    data = \
        (obj_values[TEACHER] - obj_values[LEITNER]) / obj_values[LEITNER] * 100
    # data[data[:] < 0] = 0
    # print(np.min(data))
    phase_diagram(parameter_values=parameter_values,
                  param_names=param_labels,
                  data=data,
                  fig_folder=FIG_FOLDER,
                  fig_name=f'phase_diagram_teacher_better.pdf',
                  levels=np.arange(0,
                                     np.max(data) + 10, 10))

    # data = obj_values[TEACHER]-obj_values[LEITNER]
    # data[data[:] < 0] = 0
    #
    # phase_diagram(parameter_values=parameter_values,
    #               param_names=param_labels,
    #               data=data,
    #               fig_folder=FIG_FOLDER,
    #               fig_name=f'phase_diagram_leitner_better.pdf')


# %%

if __name__ == "__main__":
    main_comparative_advantage_n_days()
