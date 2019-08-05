import matplotlib.pyplot as plt

from learner.act_r_custom import ActRMeaning
from teacher.random import RandomTeacher
from teacher.leitner import LeitnerTeacher
from teacher.complete_leitner import TraditionalLeitnerTeacher
from teacher.avya import AvyaTeacher

from simulation.memory import p_recall_over_time_after_learning

import plot.memory_trace
import plot.n_seen
import plot.n_learnt
import plot.success

from plot.generic import save_fig

from utils.utils import dic2string


def main(t_max=300, n_item=30, grades=(1, ),
         student_model=None,
         student_param=None,
         teacher_model=None, verbose=False,
         normalize_similarity=True):

    if student_model is None:
        student_model = ActRMeaning

    if teacher_model is None:
        teacher_model = RandomTeacher

    teacher = teacher_model(
        t_max=t_max, n_item=n_item, grades=grades,
        handle_similarities=True,
        normalize_similarity=normalize_similarity,
        verbose=verbose)

    learner = student_model(
        param=student_param,
        tk=teacher.tk)

    questions, replies, successes = teacher.teach(agent=learner)

    seen = teacher.seen

    # Compute the probability of recall over time
    p_recall = p_recall_over_time_after_learning(
        agent=learner,
        t_max=t_max,
        n_item=n_item)

    # Plot...
    font_size = 10
    label_size = 8
    line_width = 1

    n_rows, n_cols = 5, 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6, 14))

    ax1 = axes[0]
    plot.memory_trace.summarize(
        p_recall=p_recall,
        ax=ax1,
        font_size=font_size,
        label_size=label_size,
        line_width=line_width,
    )

    ax2 = axes[1]
    plot.memory_trace.summarize_over_seen(
        p_recall=p_recall,
        seen=seen,
        ax=ax2,
        font_size=font_size,
        label_size=label_size,
        line_width=line_width
    )

    ax3 = axes[2]
    plot.n_learnt.curve(
        p_recall=p_recall,
        ax=ax3,
        font_size=font_size,
        label_size=label_size,
        line_width=line_width
    )

    ax4 = axes[3]
    plot.n_seen.curve(
        seen=seen,
        ax=ax4,
        font_size=font_size,
        label_size=label_size,
        line_width=line_width * 2
    )

    ax5 = axes[4]
    plot.success.curve(
        successes=successes,
        ax=ax5,
        font_size=font_size,
        label_size=label_size,
        line_width=line_width * 2
    )

    extension = f'{teacher_model.__name__}_{student_model.__name__}_' \
                f'{dic2string(student_param)}_' \
                f'ni_{n_item}_grade_{grades}_tmax_{t_max}_' \
                f'norm_{normalize_similarity}'

    save_fig(f"simulation_{extension}.pdf")


if __name__ == "__main__":

    # for tm in (TraditionalLeitnerTeacher,
    #            LeitnerTeacher,
    #            RandomTeacher,
    #            AvyaTeacher):
    #     main(teacher_model=tm, t_max=1000, n_item=30,
    #          normalize_similarity=True)
    main(
        student_model=ActRMeaning,
        student_param={"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02},
        teacher_model=TraditionalLeitnerTeacher,
        t_max=2000, n_item=60, normalize_similarity=True)
