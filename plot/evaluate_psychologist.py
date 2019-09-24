import os

import matplotlib.pyplot as plt

from . generic import save_fig


SCRIPT_NAME = os.path.basename(__file__).split(".")[0]


def plot(results, extension):

    r_keys = list(results[0].keys())
    param_names = results[0][r_keys[0]].keys()

    data = {
        pn:
            {
                k: [] for k in r_keys
            }
        for pn in param_names
    }

    for r in results:
        for k in r_keys:
            for pn in param_names:
                data[pn][k].append(r[k][pn])

    n_subplot = len(r_keys)

    fig, axes = plt.subplots(nrows=n_subplot, figsize=(5, 0.9 * n_subplot))

    for i in range(n_subplot):

        color = 'black'  # f'C{i}'
        ax = axes[i]

        ax.set_ylabel('Error')
        # ax.set_yticks((0, 1))
        # ax.set_ylim((-0.1, 1.1))

        # ax.scatter(x=success_time[questions == item],
        #            y=success_value[questions == item],
        #            alpha=0.2,
        #            color=color)

        ax.plot(data[i], alpha=1, color=color)
        if i != n_subplot - 1:
            ax.set_xticks([])

    axes[-1].set_xlabel('Time')

    plt.tight_layout()

    f_name = f"{extension}.pdf"
    save_fig(fig_name=f_name, sub_folder=SCRIPT_NAME)