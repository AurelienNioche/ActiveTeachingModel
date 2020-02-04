import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "ActiveTeachingModel.settings")
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

import numpy as np
from multiprocessing import Pool

from model.teacher.leitner import Leitner

from model.simplified.scenario import run_n_session
from model.simplified.teacher import Teacher
from model.simplified.learner import ExponentialForgetting

from utils.string import dic2string


from model.plot import \
    fig_parameter_recovery, \
    fig_p_recall, fig_n_seen, fig_p_item_seen

from model.constants import \
    POST_MEAN, POST_SD, P_SEEN, N_SEEN, N_LEARNT, P_ITEM


EPS = np.finfo(np.float).eps
FIG_FOLDER = os.path.join("fig", os.path.basename(__file__))
os.makedirs(FIG_FOLDER, exist_ok=True)


def basic():

    n_iteration_per_session = 150

    sec_per_iter = 2
    n_iteration_between_session = \
        int((60 ** 2 * 24) / sec_per_iter - n_iteration_per_session)

    run_n_session(
        n_session=1,
        n_iteration_per_session=n_iteration_per_session,
        n_iteration_between_session=n_iteration_between_session,
        learner_model=ExponentialForgetting,
        teacher_model=Teacher,
        bounds=((0.001, 0.04), (0.2, 0.5)),
        param_labels=("alpha", "beta"),
        param=(0.02, 0.2),
    )


def _run_n_session(kwargs):
    return run_n_session(**kwargs)


def main_single():

    param = (0.02, 0.2)

    seed = 2
    n_iteration_per_session = 150
    sec_per_iter = 2
    n_iteration_between_session = \
        int((60 ** 2 * 24) / sec_per_iter - n_iteration_per_session)
    n_session = 1
    n_item = 1000

    grid_size = 20

    learner = ExponentialForgetting
    bounds = (0.001, 0.04), (0.2, 0.5),
    param_labels = "alpha", "beta",

    learnt_thr = 0.8

    teacher_models = Teacher, Leitner

    kwargs_list = [{
        "learner_model": learner,
        "teacher_model": teacher_model,
        "n_session": n_session,
        "n_item": n_item,
        "grid_size": grid_size,
        "param": param,
        "seed": seed,
        "n_iteration_per_session": n_iteration_per_session,
        "n_iteration_between_session": n_iteration_between_session,
        "bounds": bounds,
        "param_labels": param_labels
    } for teacher_model in teacher_models]

    sim_entries = Pool(processes=os.cpu_count())\
        .map(_run_n_session, kwargs_list)

    n_iteration = n_iteration_per_session * n_session

    timesteps = np.arange(0, n_iteration, n_iteration_per_session)

    data_type = (P_SEEN, N_SEEN, N_LEARNT, P_ITEM, POST_MEAN, POST_SD)
    data = {dt: {} for dt in data_type}

    condition_labels = [m.__name__ for m in teacher_models]

    for i, cd in enumerate(condition_labels):

        e = sim_entries[i]

        timestamps = np.array(e.timestamp, dtype=int)
        hist = np.array(e.hist, dtype=int)

        post_entries = sim_entries[i].post_set.all()

        post_mean = {}
        post_sd = {}

        for pr in param_labels:
            post_entry_pr = post_entries.get(param_label=pr)
            post_mean[pr] = np.array(post_entry_pr.mean)
            post_sd[pr] = np.array(post_entry_pr.std)

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
        f"{n_session}session" \
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

    main_single()
