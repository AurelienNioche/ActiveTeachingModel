import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "ActiveTeachingModel.settings")
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

import os
import numpy as np
from tqdm import tqdm

import pickle

from model.learner import ExponentialForgetting
from model.teacher import Teacher

from simulation_data.models import Simulation

from plot.parameter_recovery_multi import \
    fig_parameter_recovery_multi, fig_parameter_recovery_curve_multi

EPS = np.finfo(np.float).eps
FIG_FOLDER = os.path.join("fig", os.path.basename(__file__).split(".")[0])
os.makedirs(FIG_FOLDER, exist_ok=True)

BOUNDS = (0.001, 0.04), (0.2, 0.5),
PARAM_LABELS = "alpha", "beta",


def get_data():
    force = False

    bkp_file = os.path.join("bkp", "data_fig_param_recovery_grid.p")

    if not os.path.exists(bkp_file) or force:

        seed = 2
        n_iteration_per_session = 150
        sec_per_iter = 2
        n_iteration_between_session = \
            int((60 ** 2 * 24) / sec_per_iter - n_iteration_per_session)
        n_session = 60
        n_item = 1000

        grid_size = 20

        entries = Simulation.objects.filter(
            n_item=n_item,
            n_session=n_session,
            n_iteration_per_session=n_iteration_per_session,
            n_iteration_between_session=n_iteration_between_session,
            grid_size=grid_size,
            teacher_model=Teacher.__name__,
            learner_model=ExponentialForgetting.__name__,
            param_labels=PARAM_LABELS,
            param_upper_bounds=[b[0] for b in BOUNDS],
            param_lower_bounds=[b[1] for b in BOUNDS],
            seed=seed)

        n_param = len(PARAM_LABELS)

        parameter_values = np.atleast_2d([
                    np.linspace(
                        *BOUNDS[i],
                        grid_size) for i in range(n_param)
        ])

        selected_entries = []

        alpha_list = parameter_values[0, :]
        beta_list = parameter_values[1, :]

        print("Initial count", entries.count())

        for i, e in enumerate(entries):
            if np.sum(alpha_list[:] == e.param_values[0]) and np.sum(beta_list[:] == e.param_values[1]):
                selected_entries.append(e)

        n_sim = len(selected_entries)
        print("Final count", n_sim)

        obs_point = np.arange(n_iteration_per_session-1,
                              n_iteration_per_session*n_session,
                              n_iteration_per_session)
        n_obs_point = len(obs_point)

        print("N obs", n_obs_point)
        data = np.zeros((n_obs_point, n_param, 2, n_sim))

        for (i, e) in tqdm(enumerate(selected_entries), total=n_sim):
            post = e.post["mean"]
            for j, pl in enumerate(PARAM_LABELS):
                for k, obs in enumerate(obs_point):
                    data[k, j, 1, i] = post[pl][obs]

                data[:, j, 0, i] = e.param_values[j]

        os.makedirs(os.path.dirname(bkp_file), exist_ok=True)
        with open(bkp_file, "wb") as f:
            pickle.dump(data, f)

    else:
        with open(bkp_file, "rb") as f:
            data = pickle.load(f)

    data = data[:56]
    return data


def create_fig_parameter_recovery_multi():

    data = get_data()
    fig_parameter_recovery_multi(data=data, bounds=BOUNDS,
                                 fig_folder=FIG_FOLDER)


def create_fig_parameter_recovery_multi_curve():

    data = get_data()
    fig_parameter_recovery_curve_multi(
        data=data, param_labels=PARAM_LABELS,
        fig_folder=FIG_FOLDER
    )


if __name__ == "__main__":

    create_fig_parameter_recovery_multi()
    create_fig_parameter_recovery_multi_curve()
