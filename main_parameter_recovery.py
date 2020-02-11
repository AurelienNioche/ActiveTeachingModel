import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                      "ActiveTeachingModel.settings")
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from utils.plot import save_fig
import pickle
from itertools import product

from utils.multiprocessing import MultiProcess

from model.learner import ExponentialForgetting
from model.teacher import Teacher, Leitner

from simulation_data.models import Simulation

from plot.comparison import phase_diagram
from plot.correlation import fig_correlation
from plot.parameter_recovery import parameter_recovery_grid

from model.run import run_n_session

EPS = np.finfo(np.float).eps
FIG_FOLDER = os.path.join("fig", os.path.basename(__file__))


def main():
    force = False

    bkp_file = os.path.join("bkp", "data_fig_param_recovery_grid.p")

    bounds = (0.001, 0.04), (0.2, 0.5),
    param_labels = "alpha", "beta",

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
            param_labels=param_labels,
            param_upper_bounds=[b[0] for b in bounds],
            param_lower_bounds=[b[1] for b in bounds],
            seed=seed)

        n_param = len(param_labels)

        parameter_values = np.atleast_2d([
                    np.linspace(
                        *bounds[i],
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
            for j, pl in enumerate(param_labels):
                for k, obs in enumerate(obs_point):
                    data[k, j, 1, i] = post[pl][obs]

                data[:, j, 0, i] = e.param_values[j]

        os.makedirs(os.path.dirname(bkp_file), exist_ok=True)
        with open(bkp_file, "wb") as f:
            pickle.dump(data, f)

    else:
        with open(bkp_file, "rb") as f:
            data = pickle.load(f)

    # fig, axes = plt.subplots(ncols=18, nrows=7, figsize=(20, 20))

    data = data[:56]

    n_obs_point = len(data)
    print("n obs point", n_obs_point)

    n_param = data.shape[1]
    print("n_param", n_param)

    colors = [f'C{i}' for i in range(n_param)]
    alpha = 0.2
    dot_size = 2

    fig = plt.figure(figsize=(7*2, 8), constrained_layout=False)

    # gridspec inside gridspec
    outer_grid = fig.add_gridspec(1, 2, wspace=0.1, hspace=0.0,
                                  left=0.04, right=0.99, top=0.99, bottom=0.05)

    for i_param in range(2):
        inner_grid = outer_grid[i_param]\
            .subgridspec(8, 7, wspace=0.1, hspace=0.04)

        for i_obs in range(n_obs_point):
            ax = fig.add_subplot(inner_grid[i_obs])

            x = data[i_obs, i_param, 0]
            y = data[i_obs, i_param, 1]
            ax.scatter(x, y, alpha=alpha, color=colors[i_param], s=dot_size)

            # Plot identity function
            ax.plot(
                bounds[i_param],
                bounds[i_param],
                linestyle="--", alpha=0.2, color="black", zorder=-10)

            ax.set_xlim(*bounds[i_param])
            ax.set_ylim(*bounds[i_param])

            # ax.set_xticks([])
            # ax.set_yticks([])

            # Set ticks positions
            ticks = (
                bounds[i_param][0],
                # (param_bounds[i][1]-param_bounds[i][0])/2,
                bounds[i_param][1],
            )

            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            ax.set_aspect(1)

            ax.tick_params(axis='both', labelsize=5)
            ax.tick_params(axis='x', rotation=45)

            ax.set_xlabel("Simulated", fontsize=6)
            ax.set_ylabel("Recovered", fontsize=6)

            fig.add_subplot(ax)

    # k = 0
    # for row, column in product(np.arange(7), np.arange(0, 18, 2)):
    #     print(row, column, k)
    #
    #     parameter_recovery_grid(axes=axes[row, column:column+2], data=data[k],
    #                             param_labels=param_labels, param_bounds=bounds)
    #
    #     if k < len(data)-1:
    #         k += 1
    #     else:
    #         fig.delaxes(axes[row, column])
    #         fig.delaxes(axes[row, column+1])

    for ax in fig.get_axes():
        lastrow = ax.is_last_row()
        firstcol = ax.is_first_col()
        if not lastrow:
            for label in ax.get_xticklabels(which="both"):
                label.set_visible(False)
            ax.get_xaxis().get_offset_text().set_visible(False)
            ax.set_xlabel("")
            ax.tick_params(axis='x', which='both', length=0)

        if not firstcol:
            for label in ax.get_yticklabels(which="both"):
                label.set_visible(False)
            ax.get_yaxis().get_offset_text().set_visible(False)
            ax.set_ylabel("")
            ax.tick_params(axis='y', which='both', length=0)

    save_fig(fig_folder=FIG_FOLDER, fig_name=f"param_recovery_grid.pdf",
             tight_layout=False)

    # param_v = [e.param_values for e in entries if (e.param_values[0] in list(parameter_values[0, :]) and e.param_values[1] in list(parameter_values[1, :]))]
    # print(len(np.unique(param_v)))



if __name__ == "__main__":

    main()
