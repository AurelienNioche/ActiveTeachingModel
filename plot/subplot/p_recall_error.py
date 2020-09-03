from itertools import product
from typing import Hashable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from plot import utils


def plot(
    teachers: Hashable,
    learners: Hashable,
    psychologists: Hashable,
    df: pd.DataFrame,
    fig_path: str,
) -> None:

    if not isinstance(teachers, set):
        teachers = set(teachers)
    if not isinstance(learners, set):
        learners = set(learners)
    if not isinstance(psychologists, set):
        psychologists = set(psychologists)

    print("Plotting p recall error...")

    # Plot parameters
    alpha_fill_between = 0.3

    padding_0 = 0.03

    coord_min = padding_0
    coord_max = 0.5

    # Make the combinations of learners with teachers without Leitner
    learners_teachers_combos_no_leitner = set(
        product(learners,
                teachers.symmetric_difference(frozenset({"leitner"})))
    )

    p_recall_error, axes = plt.subplots(
        len(learners_teachers_combos_no_leitner),
        len(psychologists),
        sharex='all',
        sharey='all',
        figsize=(10, 10))

    # Text positions
    n_row = len(learners_teachers_combos_no_leitner)
    n_col = len(psychologists)

    # Colors
    color_mapping = utils.map_teacher_colors()

    for i_row, lt_combo in enumerate(learners_teachers_combos_no_leitner):
        for i_col, psychologist in enumerate(psychologists):

            # Text psychologist name to column
            # Text learner and teacher combo to row

            learner = next(iter(learners.intersection(lt_combo)))
            teacher = next(iter(teachers.intersection(lt_combo)))

            is_psy = df["Psychologist"] == psychologist
            is_t = df["Teacher"] == teacher
            is_learner = df["Learner"] == learner

            stacked_y = df[is_psy & is_t & is_learner]
            y = [df0["p recall error"]
                 for _, df0 in stacked_y.groupby("Agent ID")]
            y = np.asarray(y)

            mean_y = np.mean(y, axis=0)
            std_y = np.std(y, axis=0)

            color = color_mapping[teacher]

            if n_row > 1 and n_col > 1:
                ax = axes[i_row, i_col]
            elif n_row > 1:
                ax = axes[i_row]
            else:
                ax = axes[i_col]

            ax.plot(mean_y, color=color)

            ax.fill_between(
                np.arange(len(mean_y)),
                mean_y+std_y,
                mean_y-std_y,
                alpha=alpha_fill_between,
                color=color,
            )

            ax.set_ylim((0, 1))

            if i_row == 0:
                ax.set_title(psychologist)

            elif i_row == n_row:
                ax.set_xlabel("Time")

            if i_col == 0:
                ax.set_ylabel(f"{learner}, {teacher}")

    # Text left
    p_recall_error.text(
        coord_min,
        coord_max,
        "Items learnt",
        va="center",
        rotation="vertical",
        ha="center",
        transform=p_recall_error.transFigure,
    )
    # Text bottom
    p_recall_error.text(
        coord_max + 0.04,
        coord_min - 0.02,
        "Time",
        va="center",
        rotation="horizontal",
        ha="center",
        transform=p_recall_error.transFigure)

    plt.tight_layout(rect=(padding_0, 0, 1, 1))

    print("Saving fig...")
    plt.savefig(fig_path)
    print("Done!")
