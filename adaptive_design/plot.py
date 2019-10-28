import os

import matplotlib.pyplot as plt


def create_fig(param, design_types, post_means, post_sds, true_param,
               num_trial, fig_name):

    fig, axes = plt.subplots(ncols=len(param), figsize=(12, 6))

    colors = [f'C{i}' for i in range(len(param))]

    for i, ax in enumerate(axes):

        for j, dt in enumerate(design_types):

            pr = param[i]

            means = post_means[dt][pr]
            stds = post_sds[dt][pr]

            true_p = true_param[pr]
            ax.axhline(true_p, linestyle='--', color='black',
                       alpha=.2)

            ax.plot(means, color=colors[j], label=dt)
            ax.fill_between(range(num_trial), means-stds,
                            means+stds, alpha=.2, color=colors[j])

            ax.set_title(pr)

    plt.legend()
    FIG_FOLDER = os.path.join("fig", "adaptive")
    os.makedirs(FIG_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(FIG_FOLDER, fig_name))
