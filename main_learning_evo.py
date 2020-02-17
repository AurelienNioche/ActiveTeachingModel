import os

import numpy as np
import matplotlib.pyplot as plt
import string
import pickle

from tqdm import tqdm

from utils.plot import save_fig

FIG_FOLDER = os.path.join("fig", os.path.basename(__file__))

from main_grid import main_grid, PARAM_LABELS, PARAM_GRID, TEACHER_MODELS, \
    LEARNT_THR, N_ITERATION_PER_SESSION, N_SESSION, LEARNER_MODEL
from model.compute.objective import objective


def main():
    force = False

    bkp_file = os.path.join("bkp", f"data_{os.path.basename(__file__)}.p")

    n = len(PARAM_GRID)

    if not os.path.exists(bkp_file) or force:

        sim_entries = main_grid()

        obs_point = np.arange(N_ITERATION_PER_SESSION - 1,
                              N_ITERATION_PER_SESSION * N_SESSION,
                              N_ITERATION_PER_SESSION)
        n_obs_point = len(obs_point)

        data = {}

        j = 0
        for teacher_model in TEACHER_MODELS:
            t_name = teacher_model.__name__
            data[t_name] = np.zeros((n, n_obs_point))
            for i in tqdm(range(n)):

                e = sim_entries[j]
                param = e.param_values
                timestamps = e.timestamp_array
                hist = e.hist_array

                assert e.learner_model == LEARNER_MODEL.__name__

                for k, it in enumerate(obs_point):

                    t = timestamps[it]

                    p_seen = LEARNER_MODEL.p_seen_at_t(
                        hist=hist[:it+1],
                        timestamps=timestamps[:it+1],
                        param=param,
                        t=t
                    )

                    v = np.sum(p_seen[:] > LEARNT_THR)

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

    colors = [f'C{i}' for i in range(len(TEACHER_MODELS))]

    fig, ax = plt.subplots(figsize=(12, 5))

    for t_idx, teacher_model in enumerate(TEACHER_MODELS):
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