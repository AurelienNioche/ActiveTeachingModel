import os

from model.compute.objective import objective

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
from model.teacher import Teacher, Leitner, TeacherPerfectInfo

from plot.comparison import phase_diagram
from plot.correlation import fig_correlation

from model.run import run_n_session


EPS = np.finfo(np.float).eps
FIG_FOLDER = os.path.join("fig", os.path.basename(__file__))

SEED = 2
N_ITERATION_PER_SESSION = 150
SEC_PER_ITER = 2
N_ITERATION_BETWEEN_SESSION = \
    int((60 ** 2 * 24) / SEC_PER_ITER - N_ITERATION_PER_SESSION)
N_SESSION = 60
N_ITEM = 1000

GRID_SIZE = 20

LEARNER_MODEL = ExponentialForgetting
BOUNDS = (0.001, 0.04), (0.2, 0.5),
PARAM_LABELS = "alpha", "beta",

LEARNT_THR = 0.8

TEACHER_MODELS = TeacherPerfectInfo  # Teacher, Leitner,

N_PARAM = len(BOUNDS)

PARAMETER_VALUES = np.atleast_2d([
    np.linspace(
        *BOUNDS[i],
        GRID_SIZE) for i in range(N_PARAM)
])

PARAM_GRID = np.asarray(list(
    product(*PARAMETER_VALUES)
))


def make_figures():
    def truncate_10(x):
        return int(x / 10) * 10

    sim_entries = main_grid()

    n = len(PARAM_GRID)

    data = {}

    j = 0
    for teacher_model in TEACHER_MODELS:
        t_name = teacher_model.__name__
        data[t_name] = np.zeros(n)
        for i in range(n):
            data[t_name][i] = objective(sim_entries[j], LEARNT_THR)
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

    coord_alpha_x, coord_beta_x = PARAM_GRID.T

    fig_ext = f"_{N_SESSION}session_seed{SEED}"

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
    phase_diagram(parameter_values=PARAMETER_VALUES,
                  param_names=PARAM_LABELS,
                  data=data_obj,
                  fig_folder=FIG_FOLDER,
                  fig_name=
                  f'phase_diagram_teacher_better{fig_ext}.pdf',
                  levels=np.arange(truncate_10(np.min(data_obj)),
                                   np.max(data_obj) + 10, 10))


def main_grid():

    constant_param = {
        "learner_model": LEARNER_MODEL,
        "n_session": N_SESSION,
        "n_item": N_ITEM,
        "grid_size": GRID_SIZE,
        "seed": SEED,
        "n_iteration_per_session": N_ITERATION_PER_SESSION,
        "n_iteration_between_session": N_ITERATION_BETWEEN_SESSION,
        "bounds": BOUNDS,
        "param_labels": PARAM_LABELS,
    }

    kwargs_list = []
    for teacher_model in TEACHER_MODELS:
        for param in PARAM_GRID:
            kwargs_list.append({
                ** constant_param,
                ** {
                    "teacher_model": teacher_model,
                    "param": param,
                }})

    with MultiProcess(n_worker=os.cpu_count()-2) as mp:
        sim_entries = mp.map(run_n_session, kwargs_list)
    return sim_entries


if __name__ == "__main__":

    main_grid()
