import matplotlib.pyplot as plt

from utils.plot import save_fig


def fig_parameter_recovery(param, condition_labels, post_means, post_sds,
                           true_param,
                           num_trial, fig_name=None, colors=None,
                           fig_folder=None):

    fig, axes = plt.subplots(ncols=len(param), figsize=(12, 6))

    if colors is None:
        colors = [f'C{i}' for i in range(len(condition_labels))]

    for i, ax in enumerate(axes):

        for j, dt in enumerate(condition_labels):

            pr = param[i]

            means = post_means[dt][pr]
            stds = post_sds[dt][pr]

            true_p = true_param[pr]
            ax.axhline(true_p, linestyle='--', color='black',
                       alpha=.2)

            ax.plot(means, color=colors[j], label=dt)
            ax.fill_between(range(num_trial),
                            means-stds,
                            means+stds, alpha=.2, color=colors[j])

            ax.set_title(pr)
            ax.set_xlabel("Time")
            ax.set_ylabel(f"Value")

    plt.legend(loc='upper right')

    if fig_name is not None and fig_folder is not None:
        save_fig(fig_folder=fig_folder, fig_name=fig_name)

    else:
        plt.show()
