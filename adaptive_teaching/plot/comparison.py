import matplotlib.pyplot as plt
import numpy as np

from utils.plot import save_fig


def set_aspect_ratio(ax, ratio=1):

    ax.set_aspect(1.0 / ax.get_data_ratio() * ratio)


def objective_value(
        data,
        parameter_values,
        true_params,
        param_names):

    # Extract from data...
    n_param, grid_size = parameter_values.shape
    y = data.reshape([grid_size for _ in range(n_param)])

    # Create figures
    n_rows = n_param
    fig, axes = plt.subplots(nrows=n_rows, figsize=(4, 3*n_rows))

    axes[0].set_title("Parameter space exploration")

    for i in range(n_param):

        # Select relevant data
        ax = axes[i]
        param_name = param_names[i]
        x = parameter_values[i]

        # Compute mean and std
        axis = list(range(n_param))
        axis.remove(i)

        mean = np.mean(y, axis=tuple(axis))
        std = np.std(y, axis=tuple(axis))

        # Plot the mean
        ax.plot(x, mean)

        # Draw the area mean-STD, mean+STD
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.2
        )

        ax.axvline(x=true_params[i], color='red')

        ax.set_xlabel(f"{param_name}")
        ax.set_ylabel("Likelihood")

    plt.tight_layout()
    plt.show()


def phase_diagram(
        data,
        parameter_values,
        param_names,
        n_levels=200,
        title=None,
        fig_folder=None,
        fig_name=None
):

    # Extract from data...
    n_param, grid_size = parameter_values.shape
    assert n_param == 2, \
        'This figure is made for models with exactly 2 parameters!'

    x, y = parameter_values
    z = data.reshape((grid_size, grid_size)).T
    x_label, y_label = param_names

    # Create figures
    fig, ax = plt.subplots(figsize=(5, 5))

    # Axes labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Title
    ax.set_title(title)

    # Get coordinates
    x_coordinates, y_coordinates = np.meshgrid(x, y)

    # Draw phase diagram
    c = ax.contourf(x_coordinates, y_coordinates, z,
                    levels=n_levels, cmap='viridis')

    # ax.scatter(true_params[0], true_params[1], color='red')

    # Square aspect
    set_aspect_ratio(ax, 1)

    c_bar = fig.colorbar(c, ax=ax, aspect=20, shrink=0.635)
    c_bar.ax.set_ylabel('Objective value')

    plt.tight_layout()

    if fig_folder is not None and fig_name is not None:
        save_fig(fig_folder=fig_folder, fig_name=fig_name)
    else:
        plt.show()
