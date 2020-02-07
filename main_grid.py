import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "ActiveTeachingModel.settings")
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

import os
import numpy as np
from itertools import product

from utils.multiprocessing import MultiProcess

from model.learner import ExponentialForgetting
from model.teacher import Teacher, Leitner

from model.plot.comparison import phase_diagram
from model.plot.correlation import fig_correlation

from model.run import run_n_session

from simulation_data.models import Simulation

EPS = np.finfo(np.float).eps
FIG_FOLDER = os.path.join("fig", os.path.basename(__file__))


def truncate_10(x):
    return int(x/10) * 10


def objective(e, learnt_thr):

    param = e.param_values
    timestamps = np.array(e.timestamp, dtype=int)
    hist = np.array(e.hist, dtype=int)

    t = timestamps[-1]

    p_seen = eval(e.learner_model).p_seen_at_t(
        hist=hist, timestamps=timestamps, param=param, t=t
    )

    return np.sum(p_seen[:] > learnt_thr)


def main_comparative_advantage_n_session():

    seed = 0
    n_iteration_per_session = 150
    sec_per_iter = 2
    n_iteration_between_session = \
        int((60 ** 2 * 24) / sec_per_iter - n_iteration_per_session)
    n_session = 30
    n_item = 1000

    grid_size = 20

    learner_model = ExponentialForgetting.__name__
    bounds = (0.001, 0.04), (0.2, 0.5),
    param_labels = "alpha", "beta",

    learnt_thr = 0.8

    teacher_models = Teacher.__name__, Leitner.__name__

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

    kwargs_list = [{
        ** constant_param,
        ** {
            "teacher_model": teacher_model,
            "param": param,
        }} for teacher_model in teacher_models for param in param_grid]

    with MultiProcess(n_worker=os.cpu_count()-2) as mp:
        mp.map(run_n_session, kwargs_list)

    condition_labels = [m.__name__ for m in teacher_models]

    n = len(param_grid)

    data = {cd: np.zeros(n) for cd in condition_labels}

    constant_param["learner_model"] = \
        constant_param.get("learner_model").__name__

    for cd in condition_labels:

        for i, param in enumerate(param_grid):
            e = Simulation.objects.get(
                **constant_param, teacher_model=cd,
                param=list(param))
            data[cd][i] = objective(e, learnt_thr)

    data_obj = \
        (data[Teacher.__name__] - data[Leitner.__name__]) \
        / data[Leitner.__name__] * 100

    coord_alpha_x, coord_beta_x = param_grid.T

    ext = f"_{n_session}session_seed{seed}"

    fig_correlation(coord_alpha_x, data_obj,
                    x_label=r"$\alpha$",
                    y_label="Improvement (%)",
                    fig_folder=FIG_FOLDER,
                    fig_name=f'alpha_corr{ext}.pdf')
    fig_correlation(coord_beta_x, data_obj,
                    x_label=r"$\beta$",
                    y_label="Improvement (%)",
                    fig_folder=FIG_FOLDER,
                    fig_name=f'beta_corr{ext}.pdf')


    # data[data[:] < 0] = 0
    # print(np.min(data))
    phase_diagram(parameter_values=parameter_values,
                  param_names=param_labels,
                  data=data_obj,
                  fig_folder=FIG_FOLDER,
                  fig_name=
                  f'phase_diagram_teacher_better{ext}.pdf',
                  levels=np.arange(truncate_10(np.min(data_obj)),
                                   np.max(data_obj) + 10, 10))


if __name__ == "__main__":

    main_comparative_advantage_n_session()
