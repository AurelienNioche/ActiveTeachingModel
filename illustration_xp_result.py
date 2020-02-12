import matplotlib.pyplot as plt

import numpy as np
import os

from utils.plot import save_fig
from plot.scatter_metric import plot_scatter_metric
from plot.parameter_recovery_single import parameter_recovery_grid


def xp_results():
    n_subject = 30
    n_condition = 2
    data = np.zeros((n_subject, n_condition))
    data[:, 0] = np.random.normal(200, 50, size=n_subject)
    data[:, 1] = np.random.normal(300, 50, size=n_subject)
    data[data < 0] = 0
    plot_scatter_metric(data=data, y_label="N recalled",
                        x_tick_labels=["Teacher", "Leitner"],
                        fig_name="results_xp.pdf",
                        fig_folder=os.path.join("fig", "illustration"))

    fig, axes = plt.subplots(ncols=2, figsize=(5, 5))

    data = np.zeros((n_subject, 1))
    data[:, 0] = np.random.normal(0.02, 0.005, size=n_subject)
    data[data < 0] = 0.001
    plot_scatter_metric(data=data, y_label="Estimated value",
                        x_tick_labels=["$alpha$", ], ax=axes[0])

    data = np.zeros((n_subject, 1))
    data[:, 0] = np.random.normal(0.35, 0.05, size=n_subject)
    data[data < 0] = 0.001
    plot_scatter_metric(data=data, y_label="Estimated value",
                        x_tick_labels=["$beta$", ], ax=axes[1])

    save_fig(fig_name="parameter_distribution_xp.pdf",
             fig_folder=os.path.join("fig", "illustration"))


def simu_results():

    n_param = 2

    grid_size = 20
    param_bounds = ((0.001, 0.04), (0.2, 0.5))
    param_labels = ('alpha', "beta")

    data = np.zeros((n_param, grid_size, grid_size))

    for i in range(n_param):

        data[i, 0] = np.linspace(param_bounds[i][0], param_bounds[i][1],
                                 grid_size)

        noise_std = (param_bounds[i][1] - param_bounds[i][0]) * 0.5

        data[i, 1] = data[i, 0] + np.random.normal(loc=0.0, scale=noise_std, size=grid_size)

    parameter_recovery_grid(
        data,
        param_labels=param_labels,
        param_bounds=param_bounds,
        fig_name="parameter_recovery_grid.pdf",
        fig_folder=os.path.join("fig", "illustration")
    )

    # fig, axes = plt.subplots(ncols=3, figsize=(5, 15))
    #
    # data = np.zeros((n_param, grid_size, grid_size))
    #
    # noise_level = 0.5, 0.2, 0.01
    #
    # for k, nl in enumerate(noise_level)
    #
    #     for i in range(n_param):
    #
    #         data[i, 0] = np.linspace(param_bounds[i][0], param_bounds[i][1],
    #                                  grid_size)
    #
    #         noise_std = (param_bounds[i][1] - param_bounds[i][0]) * 0.1
    #
    #         data[i, 1] = data[i, 0] + np.random.normal(loc=0.0, scale=noise_std,
    #                                                    size=grid_size)
    #
    #     parameter_recovery_grid(
    #         data,
    #         axes=
    #         param_labels=param_labels,
    #         param_bounds=param_bounds)
    #
    # save_fig(fig_name="parameter_recovery_grid.pdf",
    #          fig_folder=os.path.join("fig", "illustration"))




if __name__ == "__main__":

    simu_results()
