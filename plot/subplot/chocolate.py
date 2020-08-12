import os

from itertools import combinations, product
from typing import Hashable, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import settings.paths as paths
from plot import fake_data, utils


def get_color_sequence(
    teacher_0: str,
    sequence_0: Iterable,
    teacher_1: str,
    sequence_1: Iterable,
    color_mapping: Mapping,
    ) -> tuple:
    """Get the color for each dot"""

    assert len(sequence_0) == len(sequence_1)
    epsilon = np.finfo(float).eps
    sequence_0 += epsilon
    sequence_1 += epsilon
    sequence = sequence_0 / sequence_1
    return tuple(map(lambda x: color_mapping[teacher_0] if x > 1 else color_mapping[teacher_1], sequence))


def plot(
    teachers: Hashable,
    learners: Hashable,
    psychologists: Hashable,
    df: pd.DataFrame,
    fig_path: str,
    ) -> None:
    """Prepare and save the chocolate plot"""

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

    # Start the multiscatter plot
    chocolate, axes = plt.subplots(
        len(learners_psychologists_combos),
        len(teachers_combos),
        # sharex=True,  # Uncomment for shared axes
        # sharey=True,
        figsize=(10,10)
    )

    # Plottable values per condition (models set)
    dict_cond_scores = utils.get_plot_values(df, "Agent ID", ["Teacher", "Learner", "Psychologist"], "Items learnt")

    # Text positions
    num_rows = len(learners_psychologists_combos)
    num_columns = len(teachers_combos)

    # Colors
    color_mapping = utils.map_teacher_colors()

    for n_row, learner_psychologist_combo in enumerate(learners_psychologists_combos):
        for n_col, teachers_combo in enumerate(teachers_combos):
            # Subplot parameters
            learner = next(iter(learners.intersection(learner_psychologist_combo)))
            psychologist = next(iter(psychologists.intersection(learner_psychologist_combo)))
            # Plotting data
            x = dict_cond_scores[frozenset({teachers_combo[0] , *learner_psychologist_combo})]
            y = dict_cond_scores[frozenset({teachers_combo[1] , *learner_psychologist_combo})]
            # Color
            colors = get_color_sequence(teachers_combo[0], x, teachers_combo[1], y, color_mapping)
            # Plot scatter
            axes[n_row, n_col].scatter(x, y, c=colors, alpha=alpha_dot, zorder=1)
            # Plot identity line
            axes[n_row, n_col].plot([0, 1], [0, 1], "-k", transform=axes[n_row,n_col].transAxes, alpha=alpha_line, zorder=0)
            # Label text
            # axes[n_row, n_col].set_xlabel("Items learnt " + teachers_combo[0])
            # axes[n_row, n_col].set_ylabel("Items learnt " + teachers_combo[1])

            if n_row == 0:
                axes[n_row, n_col].set_title(f"{teachers_combo[1]}, {teachers_combo[0]}")  # Inverted text indexing, easier interpretation

            elif n_row == num_rows:
                axes[n_row, n_col].set_xlabel("Time")

            if n_col == 0:
                axes[n_row, n_col].set_ylabel(f"{learner}, {psychologist}")

    # Text left
    chocolate.text(coord_min, coord_max, "Items learnt", va="center", rotation="vertical",
               ha="center", transform=chocolate.transFigure)
    # Text bottom
    chocolate.text(coord_max + 0.04, coord_min - 0.02, "Items learnt", va="center", rotation="horizontal",
               ha="center", transform=chocolate.transFigure)

    plt.tight_layout(rect=(padding_0, 0, 1, 1))

    print("Saving fig...")
    chocolate.savefig(os.path.join(fig_path, "chocolate.pdf"))
    print("Done!")

