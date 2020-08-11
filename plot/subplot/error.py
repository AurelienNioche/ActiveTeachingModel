import os

from tqdm import tqdm

from itertools import product
from typing import Hashable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_error(
    teachers: Hashable,
    learners: Hashable,
    psychologists: Hashable,
    color_mapping: dict,
    fig_path: str,
    dict_cond_scores: dict,
    ) -> None:
    """Prepare and save the chocolate plot"""

    # Make the fake data
    # teacher_colors = map_teacher_colors()

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
        figsize=(10, 10)
    )

    num_rows = len(learners_teachers_combos_no_leitner)
    # num_columns = len(psychologists)

    for i, learner_teacher_combo in tqdm(enumerate(learners_teachers_combos_no_leitner)):
        for j, psychologist in enumerate(psychologists):

            # Text psychologist name to column
            # Text learner and teacher combo to row

            # y = dict_cond_scores[frozenset({psychologist, *learner_teacher_combo})]
            y = np.random.random(10)
            y.sort()
            y = y[::-1]
            learner = next(iter(learners.intersection(learner_teacher_combo)))
            teacher = next(iter(teachers.intersection(learner_teacher_combo)))
            color = color_mapping[teacher]

            axs[i, j].plot(y, color=color)
            # mean_y = np.mean(y, axis=0)
            # std_y = np.std(y, axis=0)
            axs[i, j].fill_between(np.arange(len(y)), y * 1.3, y * 0.7,
                                   alpha=0.3, color=color)

            if i == 0:
                axs[i, j].set_title(psychologist)

            elif i == num_rows:
                axs[i, j].set_xlabel("Time")

            if j == 0:
                axs[i, j].set_ylabel(f"{learner}, {teacher}")

    # Text left
    error.text(0.03, 0.5, "Items learnt", va="center", rotation="vertical",
               ha="center", transform=error.transFigure)

    plt.tight_layout(rect=(0.03, 0, 1, 1))
    print("I'm done with creating the fig")

    print("Saving...")
    plt.savefig(os.path.join(fig_path, "error.pdf"))

