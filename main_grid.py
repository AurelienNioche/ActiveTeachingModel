import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "ActiveTeachingModel.settings")
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import string

from utils.multiprocessing import MultiProcess
from utils.plot import save_fig

from model.learner import ExponentialForgetting
from model.teacher import Teacher, Leitner

from plot.comparison import phase_diagram
from plot.correlation import fig_correlation

from model.run import run_n_session


EPS = np.finfo(np.float).eps
FIG_FOLDER = os.path.join("fig", os.path.basename(__file__))


def truncate_10(x):
    return int(x/10) * 10


def objective(e, learnt_thr):

    param = e.param_values
    timestamps = e.timestamp_array
    hist = e.hist_array

    t = timestamps[-1]

    p_seen = eval(e.learner_model).p_seen_at_t(
        hist=hist, timestamps=timestamps, param=param, t=t
    )

    return np.sum(p_seen[:] > learnt_thr)


def main_comparative_advantage_n_session():

    seed = 2
    n_iteration_per_session = 150
    sec_per_iter = 2
    n_iteration_between_session = \
        int((60 ** 2 * 24) / sec_per_iter - n_iteration_per_session)
    n_session = 60
    n_item = 1000

    grid_size = 20

    learner_model = ExponentialForgetting
    bounds = (0.001, 0.04), (0.2, 0.5),
    param_labels = "alpha", "beta",

    learnt_thr = 0.8

    teacher_models = Teacher, Leitner

    n_param = len(bounds)

    parameter_values = np.atleast_2d([
                np.linspace(
                    *bounds[i],
                    grid_size) for i in range(n_param)
    ])

    param_grid = np.asarray(list(
            product(*parameter_values)
        ))

    constant_param = {
        "learner_model": learner_model,
        "n_session": n_session,
        "n_item": n_item,
        "grid_size": grid_size,
        "seed": seed,
        "n_iteration_per_session": n_iteration_per_session,
        "n_iteration_between_session": n_iteration_between_session,
        "bounds": bounds,
        "param_labels": param_labels,
    }

    kwargs_list = []
    for teacher_model in teacher_models:
        for param in param_grid:
            kwargs_list.append({
                ** constant_param,
                ** {
                    "teacher_model": teacher_model,
                    "param": param,
                }})

    with MultiProcess(n_worker=os.cpu_count()-2) as mp:
        sim_entries = mp.map(run_n_session, kwargs_list)

    n = len(param_grid)

    data = {}

    j = 0
    for teacher_model in teacher_models:
        t_name = teacher_model.__name__
        data[t_name] = np.zeros(n)
        for i in range(n):
            data[t_name][i] = objective(sim_entries[j], learnt_thr)
            j += 1

    # data = {cd: np.zeros(n) for cd in condition_labels}

    # constant_param.pop("bounds")
    # constant_param.update({
    #       "param_upper_bounds": [b[0] for b in bounds],
    #       "param_lower_bounds": [b[1] for b in bounds],
    #       "learner_model": learner_model.__name__
    #                       })
    #
    # for cd in condition_labels:
    #
    #     for i, param in enumerate(param_grid):
    #         e = Simulation.objects.get(
    #             **constant_param,
    #             teacher_model=cd,
    #             param_values=list(param))
    #         data[cd][i] = objective(e, learnt_thr)

    data_obj = \
        (data[Teacher.__name__] - data[Leitner.__name__]) \
        / data[Leitner.__name__] * 100

    coord_alpha_x, coord_beta_x = param_grid.T

    fig_ext = f"_{n_session}session_seed{seed}"

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax = axes[0]
    fig_correlation(coord_alpha_x, data_obj,
                    ax=ax,
                    x_label=r"$\alpha$",
                    y_label="Improvement (%)")
    ax.text(-0.1, -0.1, string.ascii_uppercase[0],
            transform=ax.transAxes, size=20, weight='bold')

    ax = axes[1]
    fig_correlation(coord_beta_x, data_obj,
                    x_label=r"$\beta$",
                    y_label="Improvement (%)",
                    ax=ax)

    ax.text(-0.1, -0.1, string.ascii_uppercase[1],
            transform=ax.transAxes, size=20, weight='bold')

    save_fig(fig_name=f"relation_improvement_parameter_{fig_ext}.pdf",
             fig_folder=FIG_FOLDER)

    # data[data[:] < 0] = 0
    # print(np.min(data))
    phase_diagram(parameter_values=parameter_values,
                  param_names=param_labels,
                  data=data_obj,
                  fig_folder=FIG_FOLDER,
                  fig_name=
                  f'phase_diagram_teacher_better{fig_ext}.pdf',
                  levels=np.arange(truncate_10(np.min(data_obj)),
                                   np.max(data_obj) + 10, 10))



if __name__ == "__main__":

    main_comparative_advantage_n_session()
