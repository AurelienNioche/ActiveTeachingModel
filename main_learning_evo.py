import os

import numpy as np
import matplotlib.pyplot as plt
import string
import pickle

from tqdm import tqdm

from utils.plot import save_fig

FIG_FOLDER = os.path.join("fig", os.path.basename(__file__))

from simulation_data.models.simulation import Simulation

from model.teacher import Leitner, Teacher
from model.learner import ExponentialForgetting
from model.compute.objective import objective


def main():
    force = False

    teacher_models = (Leitner, Teacher)
    learn_thr = 0.80

    seed = 2
    n_iteration_per_session = 150
    sec_per_iter = 2
    n_iteration_between_session = \
        int((60 ** 2 * 24) / sec_per_iter - n_iteration_per_session)
    n_session = 60
    n_item = 1000

    grid_size = 20

    bounds = [(0.001, 0.04), (0.2, 0.5)]

    bkp_file = os.path.join("bkp", f"data_{os.path.basename(__file__)}.p")

    n = grid_size**len(ExponentialForgetting.param_labels)

    if not os.path.exists(bkp_file) or force:

        obs_point = np.arange(n_iteration_per_session - 1,
                              n_iteration_per_session * n_session,
                              n_iteration_per_session)
        n_obs_point = len(obs_point)

        data = {}

        j = 0
        for teacher_model in teacher_models:
            t_name = teacher_model.__name__

            sim_entries = Simulation.objects.filter(
                n_item=n_item,
                n_session=n_session,
                n_iteration_per_session=n_iteration_per_session,
                n_iteration_between_session=n_iteration_between_session,
                grid_size=grid_size,
                teacher_model=Teacher.__name__,
                learner_model=ExponentialForgetting.__name__,
                param_labels=ExponentialForgetting.param_labels,
                param_upper_bounds=[b[0] for b in bounds],
                param_lower_bounds=[b[1] for b in bounds],
                seed=seed)

            data[t_name] = np.zeros((n, n_obs_point))
            for i in tqdm(range(n)):

                e = sim_entries[j]
                param = e.param_values
                timestamps = e.timestamp_array
                hist = e.hist_array

                for k, it in enumerate(obs_point):

                    t = timestamps[it]

                    p_seen = ExponentialForgetting.p_seen_at_t(
                        hist=hist[:it+1],
                        timestamps=timestamps[:it+1],
                        param=param,
                        t=t
                    )

                    v = np.sum(p_seen[:] > learn_thr)

                    data[t_name][i, k] = v

                j += 1

        os.makedirs(os.path.dirname(bkp_file), exist_ok=True)
        with open(bkp_file, "wb") as f:
            pickle.dump(data, f)

    else:
        with open(bkp_file, "rb") as f:
            data = pickle.load(f)

    # n_obs_point, n_param, _, n_sim = data.shape
    # print("n_param", n_param)
    # print("n obs point", n_obs_point)
    # print("n sim", n_obs_point)

    colors = [f'C{i}' for i in range(len(teacher_models))]

    fig, ax = plt.subplots(figsize=(12, 5))

    for t_idx, teacher_model in enumerate(teacher_models):
        t_name = teacher_model.__name__
        for i in range(n):
            ax.plot(data[t_name][i, :],
                    alpha=0.2, linewidth=0.1, color=colors[t_idx], zorder=-10)
        ax.plot(np.mean(data[t_name], axis=0),
                label=t_name, color=colors[t_idx], linewidth=2)
            # means = np.mean(diff, axis=-1)
            # sds = np.std(diff, axis=-1)
            #
            # ax.plot(means, color=colors[i], label=lb)
            # ax.fill_between(range(n_obs_point),
            #                 means - sds,
            #                 means + sds,
            #                 alpha=.2, color=colors[i])

    ax.set_xlabel("Time")
    ax.set_ylabel("N learnt")

    ax.legend()

    save_fig(fig_name="learning_evo.pdf", fig_folder=FIG_FOLDER)


if __name__ == "__main__":

    main()