import os

import matplotlib.pyplot as plt
import pandas as pd

from utils.plot import save_fig

SCRIPT_NAME = os.path.basename(__file__).split(".")[0]


def plot(data, true_parameters, extension):

    teacher_names = list(data.keys())
    teacher_names.sort()

    data_first_teacher = data[teacher_names[0]]

    param_names = list(data_first_teacher[0].keys())
    param_names.sort()
    n_iteration = len(data_first_teacher)

    data_formatted = {}
    for tn in teacher_names:
        data_formatted[tn] = {}
        for pn in param_names:

            true_value = true_parameters[pn]
            data_param = data[tn]

            data_formatted[tn][pn] = \
                [(data_param[i][pn] - true_value)**2 for i in range(n_iteration)]

    n_subplot = len(param_names)

    fig, axes = plt.subplots(nrows=n_subplot, figsize=(10, 4 * n_subplot))

    colors = {tn: f'C{i}' for i, tn in enumerate(teacher_names)}

    for i, pn in enumerate(param_names):

        try:
            ax = axes[i]
        except TypeError:
            ax = axes
        ax.set_ylabel('Error')

        ax.axhline(0, linestyle='--', color='black', alpha=0.2)

        ax.set_title(pn)
        for tn in teacher_names:

            y = data_formatted[tn][pn]

            smooth_y = pd.Series(y).rolling(window=20).mean()

            ax.plot(y, alpha=0.1, color=colors[tn])

            ax.plot(smooth_y, alpha=0.5, color=colors[tn],
                    label=tn)

        if i != n_subplot - 1:
            ax.set_xticks([])

    plt.legend()

    try:
        axes[-1].set_xlabel('Time')
    except TypeError:
        axes.set_xlabel('Time')

    plt.tight_layout()

    f_name = f"{extension}.pdf"
    save_fig(fig_name=f_name, sub_folder=SCRIPT_NAME)
