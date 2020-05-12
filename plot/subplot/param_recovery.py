import matplotlib.pyplot as plt
import numpy as np

from utils.plot import save_fig


def fig_parameter_recovery(param_labels, cond_labels, post_means, post_sds,
                           true_param, axes=None,
                           fig_name=None, colors=None,
                           fig_folder=None):

    n_param = len(param_labels)
    n_cond = len(cond_labels)

    if axes is None:
        fig, axes = plt.subplots(ncols=n_param, figsize=(12, 6))

    if colors is None:
        colors = [f'C{i}' for i in range(n_cond)]

    for i in range(n_param):

        ax = axes[i]
        pr = param_labels[i]

        for j in range(n_cond):

            dt = cond_labels[j]

            if isinstance(post_means[j], dict):
                means = post_means[j][pr]
                stds = post_sds[j][pr]
            else:
                means = post_means[j][:, param_labels.index(pr)]
                stds = post_sds[j][:, param_labels.index(pr)]

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


def fig_parameter_recovery_heterogeneous(
        param_labels, cond_labels, post_means,
        true_param, axes=None,
        fig_name=None, colors=None,
        fig_folder=None):

    n_param = len(param_labels)
    n_cond = len(cond_labels)

    if axes is None:
        fig, axes = plt.subplots(ncols=n_param, figsize=(12, 6))

    if colors is None:
        colors = [f'C{i}' for i in range(n_cond)]

    n_item = len(post_means)

    for idx_cond in range(n_cond):

        cd = cond_labels[idx_cond]

        for idx_param in range(n_param):

            ax = axes[idx_param, idx_cond]

            for item in range(n_item):

                true_pr = true_param[item, idx_param]

                a = np.array(post_means[idx_cond][item]).T

                x = a[0]
                y = a[1+idx_param]

                y = np.abs(true_pr-y)

                ax.plot(x, y, color=colors[idx_cond], label=cd,
                        drawstyle="steps-post",
                        alpha=0.5, linewidth=0.5)

            ax.set_title(param_labels[idx_param])
            ax.set_xlabel("Time")
            ax.set_ylabel(f"Error")

            ax.legend(loc='upper right')

    if fig_name is not None and fig_folder is not None:
        save_fig(fig_folder=fig_folder, fig_name=fig_name)
