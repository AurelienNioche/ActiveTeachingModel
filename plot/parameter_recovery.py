import matplotlib.pyplot as plt

from utils.plot import save_fig


def fig_parameter_recovery(param_labels, condition_labels, post_means, post_sds,
                           true_param, axes=None,
                           fig_name=None, colors=None,
                           fig_folder=None):

    if axes is None:
        fig, axes = plt.subplots(ncols=len(param_labels), figsize=(12, 6))

    if colors is None:
        colors = [f'C{i}' for i in range(len(condition_labels))]

    for i, ax in enumerate(axes):

        for j, dt in enumerate(condition_labels):

            pr = param_labels[i]

            means = post_means[dt][pr]
            stds = post_sds[dt][pr]

            if isinstance(true_param, dict):
                true_p = true_param[pr]
            else:
                true_p = true_param[param_labels.index(pr)]
            ax.axhline(true_p, linestyle='--', color='black',
                       alpha=.2)

            ax.plot(means, color=colors[j], label=dt)
            ax.fill_between(range(len(means)),
                            means-stds,
                            means+stds, alpha=.2, color=colors[j])

            ax.set_title(pr)
            ax.set_xlabel("Time")
            ax.set_ylabel(f"Value")

        ax.legend(loc='upper right')

    if fig_name is not None and fig_folder is not None:
        save_fig(fig_folder=fig_folder, fig_name=fig_name)
    #
    # else:
    #     plt.show()


def parameter_recovery_grid(
        data,
        param_labels,
        param_bounds,
        axes=None,
        x_label='Used to simulate',
        y_label='Recovered',
        fig_name=None,
        fig_folder=None
):
    # Extract data
    n_param = len(data)

    if axes is None:
        # Create fig and axes
        n_rows, n_cols = n_param, 1
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                                 figsize=(3 * n_cols, 3 * n_rows))

    # Define colors
    colors = [f'C{i}' for i in range(n_param)]

    for i in range(n_param):

        # Select ax
        ax = axes[i]

        # Extract data
        title = param_labels[i]
        x = data[i, 0]
        y = data[i, 1]

        # Create scatter
        ax.scatter(x, y, alpha=0.5, color=colors[i])

        # Set axis label
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Set title
        ax.set_title(title)

        # Set ticks positions
        ticks = (
            param_bounds[i][0],
            (param_bounds[i][1]-param_bounds[i][0])/2,
            param_bounds[i][1],
        )

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        # Plot identity function
        ax.plot(
            param_bounds[i],
            param_bounds[i],
            linestyle="--", alpha=0.2, color="black", zorder=-10)

        # Square aspect
        ax.set_aspect(1)

    if fig_name is not None and fig_folder is not None:
        save_fig(fig_folder=fig_folder, fig_name=fig_name)