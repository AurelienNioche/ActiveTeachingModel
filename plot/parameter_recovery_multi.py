import numpy as np
import matplotlib.pyplot as plt
import string

from utils.plot import save_fig


def fig_parameter_recovery_multi(data, bounds, fig_folder,
                                 fig_name=f"param_recovery_grid.pdf"):
    n_obs_point, n_param, _, n_sim = data.shape
    print("n_param", n_param)
    print("n obs point", n_obs_point)
    print("n sim", n_obs_point)

    colors = [f'C{i}' for i in range(n_param)]
    alpha = 0.2
    dot_size = 2


    width = 12
    height = 6.5
    margin_legend = 0.52
    margin_other = 0.065
    fig = plt.figure(figsize=(width, height), constrained_layout=False)

    space = 0.4

    # gridspec inside gridspec
    outer_grid = fig.add_gridspec(1, 2, wspace=0.08, hspace=0.0,
                                  left=margin_legend/width,
                                  right=1-margin_other/width,
                                  top=1-margin_other/height,
                                  bottom=margin_legend/height)

    for i_param in range(2):
        inner_grid = outer_grid[i_param] \
            .subgridspec(8, 7, wspace=space/width,
                         hspace=(space+0.2)/height)

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

    axes = fig.get_axes()

    for ax in axes:
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

    import string

    for i, idx in enumerate((49, 105)):
        ax = axes[idx]
        ax.text(-0.55, -0.5, string.ascii_uppercase[i], transform=ax.transAxes,
                size=20, weight='bold')

    save_fig(fig_folder=fig_folder, fig_name=fig_name,
             tight_layout=False)

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


def fig_parameter_recovery_curve_multi(
        data, param_labels, fig_folder,
        fig_name=f"param_recovery_multi_curve.pdf"):

    n_obs_point, n_param, _, n_sim = data.shape
    print("n_param", n_param)
    print("n obs point", n_obs_point)
    print("n sim", n_obs_point)

    colors = [f'C{i}' for i in range(n_param)]

    fig, axes = plt.subplots(ncols=n_param, figsize=(10, 5))
    for i in range(n_param):
        diff = np.abs(data[:, i, 0, :] - data[:, i, 1, :])

        lb = param_labels[i]

        ax = axes[i]

        means = np.mean(diff, axis=-1)
        sds = np.std(diff, axis=-1)

        ax.plot(means, color=colors[i], label=lb)
        ax.fill_between(range(n_obs_point),
                        means-sds,
                        means+sds,
                        alpha=.2, color=colors[i])

        ax.set_xlabel("Time")
        ax.set_ylabel("Error")

        ax.legend()

        ax.text(-0.15, -0.1, string.ascii_uppercase[i], transform=ax.transAxes,
                size=20, weight='bold')

    save_fig(fig_name=fig_name, fig_folder=fig_folder)


# def parameter_recovery_grid(
#         data,
#         param_labels,
#         param_bounds,
#         alpha=0.2,
#         dot_size=2,
#         axes=None,
#         x_label='Used to simulate',
#         y_label='Recovered',
#         fig_name=None,
#         fig_folder=None
# ):
#     # Extract data
#     n_param = len(data)
#
#     if axes is None:
#         # Create fig and axes
#         n_rows, n_cols = n_param, 1
#         fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
#                                  figsize=(3 * n_cols, 3 * n_rows))
#
#     # Define colors
#     colors = [f'C{i}' for i in range(n_param)]
#
#     for i in range(n_param):
#         # Select ax
#         ax = axes[i]
#
#         # Extract data
#         title = param_labels[i]
#         x = data[i, 0]
#         y = data[i, 1]
#
#         # Create scatter
#         ax.scatter(x, y, alpha=alpha, color=colors[i], s=dot_size)
#
#         # Set axis label
#         ax.set_xlabel(x_label)
#         ax.set_ylabel(y_label)
#
#         # Set title
#         # ax.set_title(title)
#
#         # Set ticks positions
#         ticks = (
#             param_bounds[i][0],
#             # (param_bounds[i][1]-param_bounds[i][0])/2,
#             param_bounds[i][1],
#         )
#
#         ax.set_xticks(ticks)
#         ax.set_yticks(ticks)
#
#         # Plot identity function
#         ax.plot(
#             param_bounds[i],
#             param_bounds[i],
#             linestyle="--", alpha=0.2, color="black", zorder=-10)
#
#         # Square aspect
#         ax.set_aspect(1)
#
#     if fig_name is not None and fig_folder is not None:
#         save_fig(fig_folder=fig_folder, fig_name=fig_name,
#                  pad_inches=0)
