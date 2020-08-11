import os

from itertools import combinations, product
from typing import Hashable, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_chocolate(
    teachers: Hashable,
    learners: Hashable,
    psychologists: Hashable,
    df: pd.DataFrame,
    color_mapping: dict,
    fig_path: str,
    dict_cond_scores: dict,
    ) -> None:
    """Prepare and save the chocolate plot"""

    plt.close("all")

    # Make the combinations of all teachers
    teachers_combos = tuple(combinations(teachers, 2))

    # Make the combinations of all learners with all psychologists
    learners_psychologists_combos = tuple(product(learners, psychologists))

    # Start the multiscatter plot
    chocolate, axs = plt.subplots(
        len(learners_psychologists_combos),
        len(teachers_combos),
        sharex=True,
        sharey=True,
        #subplot_kw=dict(alpha=0.1),
        #gridspec_kw=dict(),
        figsize=(10,10)
    )

    # Colors
    def get_color_sequence(
        teacher_0: str,
        sequence_0: Iterable,
        teacher_1: str,
        sequence_1: Iterable,
        color_mapping: Mapping,
        ) -> tuple:
        """Get the color of each dot"""

        assert len(sequence_0) == len(sequence_1)
        epsilon = np.finfo(float).eps
        sequence_0 += epsilon
        sequence_1 += epsilon
        sequence = sequence_0 / sequence_1
        return tuple(map(lambda x: color_mapping[teacher_0] if x > 1 else color_mapping[teacher_1], sequence))
        #return tuple(map(lambda x: "blue" if x > 1 else "orange", sequence))

    # Text positions
    coord_min = 0.005
    coord_top = 0.989

    for i, learner_psychologist_combo in enumerate(learners_psychologists_combos):
        for j, teachers_combo in enumerate(teachers_combos):
            x = dict_cond_scores[frozenset({teachers_combo[0] , *learner_psychologist_combo})]
            y = dict_cond_scores[frozenset({teachers_combo[1] , *learner_psychologist_combo})]
            colors = get_color_sequence(teachers_combo[0], x, teachers_combo[1], y, color_mapping)

            #axs[i,j].plot( x, y, "o",)
            learner = next(iter(learners.intersection(learner_psychologist_combo)))
            psychologist = next(iter(psychologists.intersection(learner_psychologist_combo)))
            num_rows = len(learners_psychologists_combos)
            num_columns = len(teachers_combos)
            if j == 0:
                i_pos_factor = i * (1 / num_rows) + (1 / num_rows / 2)
                chocolate.text(coord_min, i_pos_factor, (learner + ", " + psychologist), va="center", rotation="vertical")
            if i == 0:
                j_pos_factor = j * (1 / num_columns) + (1 / num_columns / 2)
                chocolate.text(j_pos_factor, coord_top , teachers_combo[0] + ", " + teachers_combo[1], ha="center")
            axs[i,j].scatter(x, y, c=colors, alpha=0.9, zorder=1)
            axs[i,j].plot([0, 1], [0, 1], "-k", transform=axs[i,j].transAxes, alpha=0.5, zorder=0)
            axs[i,j].set_xlabel("Items learnt " + teachers_combo[0])
            axs[i,j].set_ylabel("Items learnt " + teachers_combo[1])

    # axs[0,0].legend()
    plt.tight_layout()
    chocolate.savefig(os.path.join(fig_path, "chocolate.pdf"))

