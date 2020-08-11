import os

from itertools import product
from typing import Hashable

import matplotlib.pyplot as plt
import pandas as pd


def plot_error(
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

    # Make the fake data
    #teacher_colors = map_teacher_colors()

    # Make the combinations of all learners with all teachers
    learners_teachers_combos_no_leitner = set(
            product(learners, teachers.symmetric_difference(frozenset({"leitner"})))
        )

    error, axs = plt.subplots(
        len(learners_teachers_combos_no_leitner),
        len(psychologists),
        sharex=True,
        sharey=True,
        #subplot_kw=dict(alpha=0.9),
        #gridspec_kw=dict(),
        figsize=(10,10)
    )

    # Text positions
    coord_min = 0.005 - 0.01
    coord_max = 0.5
    coord_top = 0.989


    for i, learner_teacher_combo in enumerate(learners_teachers_combos_no_leitner):
        for j, psychologist in enumerate(psychologists):
            y = dict_cond_scores[frozenset({psychologist, *learner_teacher_combo})]
            learner = next(iter(learners.intersection(learner_teacher_combo)))
            teacher = next(iter(teachers.intersection(learner_teacher_combo)))
            color = color_mapping[teacher]
            #print((str(learner_teacher_combo)))
            #print((str(psychologist)))
            #axs[i,j].plot( x, y, "o",)
            #axs[i,j].scatter( x, y, c=colors)
            # sns.lineplot(x, y=np.arange(len(x)), ax=axes[i, j])
            axs[i,j].plot(y, color=color)
            axs[i,j].fill_between(range(len(y)), y * 1.3, y * 0.7, alpha=0.3, color=color)
            # Text psychologist name to column
            # Text learner and teacher combo to row
            num_rows = len(learners_teachers_combos_no_leitner)
            num_columns = len(psychologists)
            if j == 0:
                i_pos_factor = i * (1 / num_rows) + (1 / num_rows / 2)
                error.text(coord_min, i_pos_factor, (learner + ", " + teacher), va="center", rotation="vertical")
            if i == 0:
                j_pos_factor = j * (1 / num_columns) + (1 / num_columns / 2)
                error.text(j_pos_factor, coord_top , psychologist, ha="center")
            #axs[i,j].plot([0, 1], [0, 1], "--k", transform=axs[i,j].transAxes)
            #axs[i,j].set_xlabel(psychologist)
            #axs[i,j].set_ylabel()
    #sns.lineplot(data=p_recall_error)
    # Text bottom
    error.text(coord_max, coord_min, "Time", ha="center")
    # Text left
    error.text(coord_min, coord_max, "Items learnt", va="center", rotation="vertical")
    plt.tight_layout()
    #plt.margins(0.9)
    error.savefig(os.path.join(fig_path, "error.pdf"))

