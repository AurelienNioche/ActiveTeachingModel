import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "ActiveTeachingModel.settings")
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

import os
import numpy as np
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool

from model.simplified.learner import ExponentialForgetting

from model.plot.comparison import phase_diagram
from model.plot.correlation import fig_correlation

from model.teacher.leitner import Leitner
from model.simplified.teacher import Teacher

from model.simplified.scenario import run_n_session
from utils.multiprocessing import MultiProcess

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


def grid_exploration_n_session(parameter_values, **kwargs):

    # Create a grid for each parameter
    param_grid = np.asarray(list(
            product(*parameter_values)
        ))

    n_sets = len(param_grid)

    kwargs_list = [{
        **kwargs,
        **{
            "param": param_grid[i],
        }
    } for i in range(n_sets)]

    with MultiProcess(n_worker=os.cpu_count()-2) as mp:
        results = mp.map(run_n_session, kwargs_list)

    # with Pool(os.cpu_count()-1) as p:
    #     results = list(tqdm(p.imap(_run_n_session, kwargs_list),
    #                    total=n_sets))

    return results


def main_comparative_advantage_n_session():

    seed = 0
    n_iteration_per_session = 150
    sec_per_iter = 2
    n_iteration_between_session = \
        int((60 ** 2 * 24) / sec_per_iter - n_iteration_per_session)
    n_session = 30
    n_item = 1000

    grid_size = 20

    learner_model = ExponentialForgetting
    bounds = (0.001, 0.04), (0.2, 0.5),
    param_labels = "alpha", "beta",

    learnt_thr = 0.8

    teacher_models = Teacher, Leitner

    kwargs_list = [{
        "learner_model": learner_model,
        "teacher_model": teacher_model,
        "n_session": n_session,
        "n_item": n_item,
        "grid_size": grid_size,
        "seed": seed,
        "n_iteration_per_session": n_iteration_per_session,
        "n_iteration_between_session": n_iteration_between_session,
        "bounds": bounds,
        "param_labels": param_labels
    } for teacher_model in teacher_models]

    condition_labels = [m.__name__ for m in teacher_models]

    n_param = len(bounds)

    parameter_values = np.atleast_2d([
                np.linspace(
                    *bounds[i],
                    grid_size) for i in range(n_param)
    ])

    results = {}

    for i, cd in enumerate(condition_labels):

        results[cd] = grid_exploration_n_session(
            parameter_values=parameter_values,
            **kwargs_list[i]
        )

        # data = obj_values[cd]

        # phase_diagram(parameter_values=parameter_values,
        #               param_names=param_labels,
        #               data=data,
        #               fig_folder=FIG_FOLDER,
        #               fig_name=f'phase_diagram_{cd}.pdf',
        #               levels=np.linspace(np.min(data), np.max(data), 10))

    data = {}

    for cd in condition_labels:

        data[cd] = np.array([objective(e, learnt_thr)
                             for e in results[cd]])

    data_obj = \
        (data[Teacher.__name__] - data[Leitner.__name__]) \
        / data[Leitner.__name__] * 100

    parameter_values_array = np.asarray(list(
            product(*parameter_values)
        ))

    coord_alpha_x, coord_beta_x = parameter_values_array.T

    # for i, (alpha, beta) in enumerate(parameter_values_array):
    #
    #     coord_alpha_x.append(alpha)
    #     coord
    #     coord_y.append(data[i])

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
