# %%
import os
import numpy as np
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool, Manager

from utils.decorator import use_pickle

from model.simplified.learner import ExponentialForgetting

from model.plot.comparison import phase_diagram
from model.simplified.scenario import run_n_days
from model.plot.correlation import fig_correlation

from model.plot import fig_parameter_recovery, \
    fig_p_recall, fig_p_recall_item, fig_n_seen, \
    fig_p_item_seen

from utils.string import dic2string

from model.constants import \
    POST_MEAN, POST_SD, \
    P, P_SEEN, FR_SEEN, N_SEEN, HIST, TIMESTAMP, OBJECTIVE, N_LEARNT, P_ITEM

EPS = np.finfo(np.float).eps
FIG_FOLDER = os.path.join("fig", "scenario")


from model.simplified.labels \
    import LEITNER, TEACHER


def objective(results,):
    p_seen = results[P_SEEN]
    return np.sum(p_seen[-1][:] > 0.80)


def truncate_10(x):
    return int(x/10) * 10


def _run_n_days(kwargs):
    r = run_n_days(**kwargs)
    return r


@use_pickle
def grid_exploration_n_days(
        parameter_values,
        bounds, grid_size, **kwargs):

    # Create a grid for each parameter
    param_grid = np.asarray(list(
            product(*parameter_values)
        ))

    n_sets = len(param_grid)

    m = Manager()
    lock = m.Lock()

    kwargs_list = [{
        **kwargs,
        **{
            "bounds": bounds,
            "grid_size": grid_size,
            "param": param_grid[i],
            "using_multiprocessing": True,
            "lock": lock,
        }
    } for i in range(n_sets)]

    with Pool(os.cpu_count()) as p:
        r = list(tqdm(p.imap(_run_n_days, kwargs_list),
                      total=len(kwargs_list)))

    return r


def main_comparative_advantage_n_days():

    seed = 1
    n_iter_session = 150
    n_day = 60
    n_item = 1000

    grid_size = 20

    learner = ExponentialForgetting
    bounds = (0.001, 0.04), (0.2, 0.5),
    param_labels = "alpha", "beta",

    condition_labels = \
        TEACHER, LEITNER  # , ADAPTIVE

    n_param = len(bounds)

    parameter_values = np.atleast_2d([
                np.linspace(
                    *bounds[i],
                    grid_size) for i in range(n_param)
    ])

    results = {}

    for cd in condition_labels:

        results[cd] = grid_exploration_n_days(
            learner=learner,
            parameter_values=parameter_values,
            condition=cd,
            n_item=n_item,
            n_iter_session=n_iter_session,
            n_day=n_day,
            bounds=bounds,
            grid_size=grid_size,
            param_labels=param_labels,
            seed=seed
        )

        # data = obj_values[cd]

        # phase_diagram(parameter_values=parameter_values,
        #               param_names=param_labels,
        #               data=data,
        #               fig_folder=FIG_FOLDER,
        #               fig_name=f'phase_diagram_{cd}.pdf',
        #               levels=np.linspace(np.min(data), np.max(data), 10))

    data_type = (POST_MEAN, POST_SD, P, P_SEEN, FR_SEEN, N_SEEN, HIST)
    data = {dt: {} for dt in data_type}
    data[OBJECTIVE] = {}

    for cd in condition_labels:
        for dt in data_type:
            d = [r[dt] for r in results[cd]]
            data[dt][cd] = d

        data[OBJECTIVE][cd] = np.array([objective(r) for r in results[cd]])

    data_obj = \
        (data[OBJECTIVE][TEACHER] - data[OBJECTIVE][LEITNER]) \
        / data[OBJECTIVE][LEITNER] * 100

    parameter_values_array = np.asarray(list(
            product(*parameter_values)
        ))

    coord_alpha_x, coord_beta_x = parameter_values_array.T

    # for i, (alpha, beta) in enumerate(parameter_values_array):
    #
    #     coord_alpha_x.append(alpha)
    #     coord
    #     coord_y.append(data[i])

    fig_correlation(coord_alpha_x, data_obj,
                    x_label=r"$\alpha$",
                    y_label="Improvement (%)",
                    fig_folder=FIG_FOLDER,
                    fig_name=f'alpha_corr_{n_day}days_seed{seed}.pdf')
    fig_correlation(coord_beta_x, data_obj,
                    x_label=r"$\beta$",
                    y_label="Improvement (%)",
                    fig_folder=FIG_FOLDER,
                    fig_name=f'beta_corr_{n_day}days_seed{seed}.pdf')


    # data[data[:] < 0] = 0
    # print(np.min(data))
    phase_diagram(parameter_values=parameter_values,
                  param_names=param_labels,
                  data=data_obj,
                  fig_folder=FIG_FOLDER,
                  fig_name=
                  f'phase_diagram_teacher_better_{n_day}days_seed{seed}.pdf',
                  levels=np.arange(truncate_10(np.min(data_obj)),
                                   np.max(data_obj) + 10, 10))

    # data = obj_values[TEACHER]-obj_values[LEITNER]
    # data[data[:] < 0] = 0
    #
    # phase_diagram(parameter_values=parameter_values,
    #               param_names=param_labels,
    #               data=data,
    #               fig_folder=FIG_FOLDER,
    #               fig_name=f'phase_diagram_leitner_better.pdf')


# def _run_n_days(kwargs):
#     return run_n_days(**kwargs)


def main_single():

    param = (0.02, 0.2)

    seed = 2
    n_iter_session = 150
    sec_per_iter = 2
    n_day = 90
    n_item = 1000

    grid_size = 20

    learner = ExponentialForgetting
    bounds = (0.001, 0.04), (0.2, 0.5),
    param_labels = "alpha", "beta",

    learnt_thr = 0.8

    condition_labels = \
        TEACHER, LEITNER

    kwargs_list = [{
        "learner": learner,
        "n_day": n_day,
        "n_item": n_item,
        "grid_size": grid_size,
        "param": param,
        "seed": seed,
        "condition": cd,
        "n_iter_session": n_iter_session,
        "sec_per_iter": sec_per_iter,
        "bounds": bounds,
        "param_labels": param_labels
    } for cd in condition_labels]

    results = Pool(processes=os.cpu_count()).map(_run_n_days, kwargs_list)

    n_iter_break = int((60**2 * 24) / sec_per_iter - n_iter_session)
    n_iter_day = n_iter_session + n_iter_break
    n_iter = n_iter_day * n_day

    timesteps = np.arange(0, n_iter, n_iter_day)

    data_type = (P_SEEN, N_SEEN, N_LEARNT, P_ITEM, POST_MEAN, POST_SD)
    data = {dt: {} for dt in data_type}

    for i, cd in enumerate(condition_labels):

        timestamps = results[i][TIMESTAMP]
        hist = results[i][HIST]
        post_mean = results[i][POST_MEAN]
        post_sd = results[i][POST_SD]

        d = learner.stats_ex_post(
            param_labels=param_labels,
            param=param, hist=hist, timestamps=timestamps,
            timesteps=timesteps, learnt_thr=learnt_thr,
            post_mean=post_mean, post_sd=post_sd
        )
        for dt in data_type:
            data[dt][cd] = d[dt]

    fig_ext = \
        "_" \
        f"{learner.__name__}_" \
        f"{dic2string(param)}_" \
        f"{n_day}days" \
        f".pdf"

    fig_name = f"p_seen" + fig_ext
    fig_p_recall(data=data[P_SEEN], condition_labels=condition_labels,
                 fig_name=fig_name, fig_folder=FIG_FOLDER)

    fig_name = f"p_item" + fig_ext
    fig_p_item_seen(
        p_recall=data[P_ITEM], condition_labels=condition_labels,
        fig_name=fig_name, fig_folder=FIG_FOLDER)

    fig_name = f"n_seen" + fig_ext
    fig_n_seen(
        data=data[N_SEEN], y_label="N seen",
        condition_labels=condition_labels,
        fig_name=fig_name, fig_folder=FIG_FOLDER)

    fig_name = f"n_learnt" + fig_ext
    fig_n_seen(
        data=data[N_LEARNT], y_label="N learnt",
        condition_labels=condition_labels,
        fig_name=fig_name, fig_folder=FIG_FOLDER)

    fig_name = f"param_recovery" + fig_ext
    fig_parameter_recovery(condition_labels=condition_labels,
                           param_labels=param_labels,
                           post_means=data[POST_MEAN], post_sds=data[POST_SD],
                           true_param=param,
                           fig_name=fig_name,
                           fig_folder=FIG_FOLDER)


if __name__ == "__main__":
    main_comparative_advantage_n_days()
