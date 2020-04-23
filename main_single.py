import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "ActiveTeachingModel.settings")
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

from simulation_data.models import Simulation

import numpy as np
import string

from model.run import run_n_session
from model.teacher import Teacher, Leitner, Psychologist, \
    TeacherPerfectInfo
from model.learner import ExponentialForgetting

from utils.string import dic2string, param_string
from utils.multiprocessing import MultiProcess

from plot import \
    fig_parameter_recovery, \
    fig_p_recall, fig_n_against_time, fig_p_item_seen

from plot.single import DataFigSingle, fig_single

from model.constants import \
    POST_MEAN, POST_SD, P_SEEN, N_SEEN, N_LEARNT, P_ITEM

import matplotlib.pyplot as plt

from utils.plot import save_fig


EPS = np.finfo(np.float).eps
FIG_FOLDER = os.path.join("fig", os.path.basename(__file__).split(".")[0])
os.makedirs(FIG_FOLDER, exist_ok=True)


def basic():

    n_iteration_per_session = 150

    sec_per_iter = 2
    n_iteration_between_session = \
        int((60 ** 2 * 24) / sec_per_iter - n_iteration_per_session)

    run_n_session(
        n_session=3,
        n_iteration_per_session=n_iteration_per_session,
        n_iteration_between_session=n_iteration_between_session,
        learner_model=ExponentialForgetting,
        teacher_model=TeacherPerfectInfo,
        bounds=((0.001, 0.04), (0.2, 0.5)),
        param_labels=("alpha", "beta"),
        param=(0.02, 0.2),
        seed=2
    )


def _run_n_session(kwargs):
    return run_n_session(**kwargs)


def main_single():

    Simulation.objects.all().delete()

    param = (0.02, 0.2)

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

    kwargs_list = [{
        "learner_model": learner_model,
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

    with MultiProcess(n_worker=os.cpu_count()) as mp:
        sim_entry_ids = mp.map(run_n_session, kwargs_list)

    # n_iteration = n_iteration_per_session * n_session

    n_time_steps = \
        (n_iteration_per_session + n_iteration_between_session) * n_session

    n_time_steps_per_day = \
        n_iteration_per_session + n_iteration_between_session

    timesteps = np.arange(0, n_time_steps, n_time_steps_per_day)

    # data_type = (P_SEEN, N_SEEN, N_LEARNT, P_ITEM, POST_MEAN, POST_SD)
    # data = {dt: {} for dt in data_type}

    condition_labels = [m.__name__ for m in teacher_models]

    data = DataFigSingle(param_labels=param_labels, param=param)

    for i, cd in enumerate(condition_labels):

        e = Simulation.objects.get(id=sim_entry_ids[i])

        timestamps = e.timestamp_array
        hist = e.hist_array

        # n_seen_array = e.n_seen_array

        # timesteps = timestamps.copy()

        # print(f"cd {cd}, timestamps {timestamps}")

        # post_entries = sim_entries[i].post_set.all()

        post = e.post

        d = learner_model().stats_ex_post(
            param_labels=param_labels,
            param=param, hist=hist, timestamps=timestamps,
            timesteps=timesteps, learnt_thr=learnt_thr,
            post=post
        )

        data.add(
            n_learnt=d.n_learnt,
            n_seen=d.n_seen,
            p_item=d.p_item,
            label=cd,
            post_mean=d.post_mean,
            post_std=d.post_std
        )

    fig_ext = \
        f"{learner_model.__name__}_" \
        f"{param_string(param_labels=param_labels, param=param)}_" \
        f"n_session={n_session}"
    fig_single(data=data, fig_folder=FIG_FOLDER, fig_name=fig_ext)


if __name__ == "__main__":

    main_single()
