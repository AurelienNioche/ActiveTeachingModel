import os

from itertools import product
from typing import Hashable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import settings.paths as paths

from plot import utils


def plot(
    teachers: Hashable,
    learners: Hashable,
    psychologists: Hashable,
    df: pd.DataFrame,
    fig_path: str,
    ) -> None:
    """Prepare and save the chocolate plot"""

    print("Plotting p recall error...")

    # Plot parameters
    alpha_fill_between = 0.3

    padding_0 = 0.03

    coord_min = padding_0
    coord_max = 0.5

    # Make the combinations of learners with teachers without Leitner
    learners_teachers_combos_no_leitner = set(
            product(learners, teachers.symmetric_difference(frozenset({"leitner"})))
        )

    p_recall_error, axes = plt.subplots(
        len(learners_teachers_combos_no_leitner),
        len(psychologists),
        sharex=True,
        sharey=True,
        figsize=(10, 10)
    )

    # Plottable values per condition (models set)
    dict_cond_scores = utils.get_plot_values(df, "Agent ID", ["Teacher", "Learner", "Psychologist"], "p recall error")

    # Text positions
    num_rows = len(learners_teachers_combos_no_leitner)
    # num_columns = len(psychologists)

    # Colors
    color_mapping = utils.map_teacher_colors()

    for n_row, learner_teacher_combo in enumerate(learners_teachers_combos_no_leitner):
        for n_col, psychologist in enumerate(psychologists):

            # Text psychologist name to column
            # Text learner and teacher combo to row

            y = dict_cond_scores[frozenset({psychologist, *learner_teacher_combo})]
            y = np.random.random(300)
            y.sort()
            y = y[::-1]
            learner = next(iter(learners.intersection(learner_teacher_combo)))
            teacher = next(iter(teachers.intersection(learner_teacher_combo)))
            color = color_mapping[teacher]

            axes[n_row, n_col].plot(y, color=color)
            # mean_y = np.mean(y, axis=0)
            # std_y = np.std(y, axis=0)
            axes[n_row, n_col].fill_between(np.arange(len(y)), y * 1.3, y * 0.7,
                                   alpha=alpha_fill_between, color=color)

            if n_row == 0:
                axes[n_row, n_col].set_title(psychologist)

            elif n_row == num_rows:
                axes[n_row, n_col].set_xlabel("Time")

            if n_col == 0:
                axes[n_row, n_col].set_ylabel(f"{learner}, {teacher}")

    # Text left
    p_recall_error.text(coord_min, coord_max, "Items learnt", va="center", rotation="vertical",
               ha="center", transform=p_recall_error.transFigure)
    # Text bottom
    p_recall_error.text(coord_max + 0.04, coord_min - 0.02, "Time", va="center", rotation="horizontal",
               ha="center", transform=p_recall_error.transFigure)

    plt.tight_layout(rect=(padding_0, 0, 1, 1))

    print("Saving fig...")
    plt.savefig(os.path.join(fig_path, "p_recall_error.pdf"))
    print("Done!")

