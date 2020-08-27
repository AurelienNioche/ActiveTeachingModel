# import os
from itertools import combinations, product
from typing import Hashable, Iterable, Mapping

import matplotlib.pyplot as plt
import pandas as pd

# import numpy as np

# import settings.paths as paths
from plot import utils  # fake_data,


def get_color_sequence(
    teacher_0: str,
    sequence_0: Iterable,
    teacher_1: str,
    sequence_1: Iterable,
    color_mapping: Mapping,
) -> tuple:
    """Get the color for each dot"""

    assert len(sequence_0) == len(sequence_1)

    sequence = sequence_0 / sequence_1

    return tuple(
        map(
            lambda x: color_mapping[teacher_0] if x > 1 else color_mapping[teacher_1],
            sequence,
        )
    )


def plot(
    learnt_label: str,
    teachers: Hashable,
    learners: Hashable,
    psychologists: Hashable,
    df: pd.DataFrame,
    fig_path: str,
) -> None:
    """Prepare and save the chocolate plot
    dataframe needs following columns:
    "Agent ID", "Teacher", "Learner", Psychologist", "Items learnt"
    """

    if not isinstance(teachers, set):
        teachers = set(teachers)
    if not isinstance(learners, set):
        learners = set(learners)
    if not isinstance(psychologists, set):
        psychologists = set(psychologists)

    print("Plotting multiscatter...")
    # Plot parameters
    alpha_dot = 0.7
    alpha_line = 0.4

    padding_0 = 0.03

    coord_min = padding_0
    coord_max = 0.5

    # Make the combinations of all teachers
    teachers_combos = tuple(combinations(teachers, 2))

    # Make the combinations of all learners with all psychologists
    learners_psychologists_combos = tuple(product(learners, psychologists))

    min_x, max_x = min(df[learnt_label]), max(df[learnt_label])

    # Start the multiscatter plot
    chocolate, axes = plt.subplots(
        len(learners_psychologists_combos),
        len(teachers_combos),
        sharex=True,  # Uncommented for shared axes
        sharey=True,
        figsize=(10, 10),
    )

    # Plottable values per condition (models set)
    dict_cond_scores = utils.get_plot_values(
        df, "Agent ID", ["Teacher", "Learner", "Psychologist"], learnt_label
    )

    # Text positions
    n_row = len(learners_psychologists_combos)
    n_col = len(teachers_combos)

    # Colors
    color_mapping = utils.map_teacher_colors()

    for i_row, learner_psychologist_combo in enumerate(learners_psychologists_combos):
        for i_col, teachers_combo in enumerate(teachers_combos):
            # Subplot parameters
            learner = next(iter(learners.intersection(learner_psychologist_combo)))
            psychologist = next(
                iter(psychologists.intersection(learner_psychologist_combo))
            )
            # Plotting data
            x = dict_cond_scores[
                frozenset({teachers_combo[0], *learner_psychologist_combo})
            ]
            y = dict_cond_scores[
                frozenset({teachers_combo[1], *learner_psychologist_combo})
            ]
            # Color
            colors = get_color_sequence(
                teachers_combo[0], x, teachers_combo[1], y, color_mapping
            )
            # Plot scatter
            if n_row > 1 and n_col > 1:
                ax = axes[i_row, i_col]
            elif n_row > 1:
                ax = axes[i_row]
            else:
                ax = axes[i_col]

            ax.scatter(x, y, c=colors, alpha=alpha_dot, zorder=1)
            # Plot identity line
            ax.plot(
                [0, 1],
                [0, 1],
                "-k",
                transform=ax.transAxes,
                alpha=alpha_line,
                zorder=0,
            )

            # Label text
            if i_row == 0:
                ax.set_title(
                    f"{teachers_combo[1]} vs. {teachers_combo[0]}"
                )  # Inverted text indexing, easier interpretation

            elif i_row == n_row:
                ax.set_xlabel("{teachers_combos[0]} \n Time")

            if i_col == 0:
                ax.set_ylabel(f"{learner}, {psychologist} \n {teachers_combo[1]}")

            ax.set_aspect(1)
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_x, max_x)

    # Text left
    # DO NOT REMOVE, FOR FINAL PLOT
    chocolate.text(
        coord_min,
        coord_max,
        "Items learnt",
        va="center",
        rotation="vertical",
        ha="center",
        transform=chocolate.transFigure,
    )
    # Text bottom
    chocolate.text(
        coord_max + 0.04,
        coord_min - 0.02,
        "Items learnt",
        va="center",
        rotation="horizontal",
        ha="center",
        transform=chocolate.transFigure,
    )

    plt.tight_layout(rect=(padding_0, 0, 1, 1))

    print("Saving fig...")
    plt.show()
    chocolate.savefig(fig_path)
    print("Done!")
