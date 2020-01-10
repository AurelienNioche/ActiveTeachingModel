import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
        vmin=None,
        vmax=None,
        levels=10,
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

    # if vmin is None:
    #     vmin = np.min(z)
    # if vmax is None:
    #     vmax = np.max(z)
    # print(vmin, vmax)
    # levels = np.linspace(vmin, vmax, n_levels)
    # else:
    #     levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
    #               130, 140, 150, 160]
    # print(levels)

    try:

        # Draw phase diagram
        c = ax.contourf(x_coordinates, y_coordinates, z, vmin=vmin, vmax=vmax,
                        levels=levels, cmap='hot')

        ax.set_aspect(0.1)
        # ax.scatter(true_params[0], true_params[1], color='red')

        # m = plt.cm.ScalarMappable(cmap=cm.get_cmap('viridis'))
        # m.set_array(z)
        # m.set_clim(0., 1000.)
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        #  divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)

        c_bar = fig.colorbar(c, ax=ax, boundaries=[vmin, vmax],
                             ticks=levels,
                             fraction=0.046, pad=0.04)
        c_bar.ax.set_ylabel('Objective value')

    except Exception:
        pass

    # Square aspect
    # set_aspect_ratio(ax, 1)

    # plt.tight_layout()

    if fig_folder is not None and fig_name is not None:
        save_fig(fig_folder=fig_folder, fig_name=fig_name)
    else:
        plt.show()
