import plot.memory_trace
import plot.success
import plot.n_seen
import plot.n_learnt

from learner.act_r_custom import ActRMeaning
from learner.half_life import HalfLife

from simulation.run import run

from teacher.active import Active
from teacher.leitner import Leitner
from teacher.random import RandomTeacher

import matplotlib.pyplot as plt
from plot.generic import save_fig

from utils.utils import dic2string

import os

SCRIPT_NAME = os.path.basename(__file__).split(".")[0]


def main(force=False):

    # Task attributes
    n_item = 150   # 30, 150
    n_iteration = 10  # 4000

    # # Student
    student_model = HalfLife
    student_param = {
        "beta": 0.02,
        "alpha": 0.2
    }
    # student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02}
    #     # {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.1}
    # # student_param = {"d": 0.5, "tau": 0.01, "s": 0.06,
    # #                  "m": 0.1}
    # student_model = ActRMeaning

    # Teacher
    teacher_models = (RandomTeacher, Leitner, Active)

    # Plot
    font_size = 8
    label_size = 6
    line_width = 1

    n_rows, n_cols = 5, len(teacher_models)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 14))

    j = 0

    for teacher_model in teacher_models:
        r = run(
            teacher_model=teacher_model,
            student_model=student_model,
            student_param=student_param,
            n_item=n_item, n_iteration=n_iteration,
            verbose=False, force=force)

        p_recall = r['p_recall']
        seen = r['seen']
        successes = r['successes']

        ax1 = axes[0, j]
        plot.memory_trace.summarize(
            p_recall=p_recall,
            ax=ax1,
            font_size=font_size,
            label_size=label_size,
            line_width=line_width,
        )

        ax2 = axes[1, j]
        plot.memory_trace.summarize_over_seen(
            p_recall=p_recall,
            seen=seen,
            ax=ax2,
            font_size=font_size,
            label_size=label_size,
            line_width=line_width
        )

        ax3 = axes[2, j]
        plot.n_learnt.curve(
            p_recall=p_recall,
            ax=ax3,
            font_size=font_size,
            label_size=label_size,
            line_width=line_width
        )

        ax4 = axes[3, j]
        plot.n_seen.curve(
            seen=seen,
            ax=ax4,
            font_size=font_size,
            label_size=label_size,
            line_width=line_width*2
        )

        ax5 = axes[4, j]
        plot.success.curve(
            successes=successes,
            ax=ax5,
            font_size=font_size,
            label_size=label_size,
            line_width=line_width*2
        )

        j += 1

    extension = \
        '_'.join([t.__name__ for t in teacher_models]) +  \
        f'{student_model.__name__}_' \
        f'{dic2string(student_param)}_' \
        f'n_item_{n_item}_n_iteration_{n_iteration}'

    save_fig(
        f'{extension}.pdf',
        sub_folder=SCRIPT_NAME
    )


if __name__ == "__main__":

    main(force=True)
