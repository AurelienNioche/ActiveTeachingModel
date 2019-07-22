import plot.memory_trace
import plot.success
import plot.n_seen
import plot.n_learnt
from learner.act_r import ActR
from learner.act_r_custom import ActRMeaning, ActRGraphic, ActRPlus
from learner.rl import QLearner
from simulation.memory import p_recall_over_time_after_learning
from teacher.avya import AvyaTeacher
# from teacher.avya_leitner import AvyaLeitTeacher
from teacher.leitner import LeitnerTeacher
from teacher.random import RandomTeacher

import matplotlib.pyplot as plt
from plot.generic import save_fig

from utils.utils import dic2string, load, dump

import os


def _produce_data(student_model, teacher_model, student_param,
                  n_item, grades, t_max, verbose=False):

    teacher = teacher_model(t_max=t_max, n_item=n_item, grades=grades)
    learner = student_model(param=student_param, tk=teacher.tk)

    print(f"\nSimulating data with a student {student_model.__name__} "
          f"(parameters={student_param}), "
          f"and a teacher {teacher_model.__name__} "
          f"using {n_item} kanji of grade {grades} for {t_max} time steps...",
          end=" ", flush=True)

    questions, replies, successes = \
        teacher.teach(agent=learner, verbose=verbose)

    print("Done.")
    print('Computing probabilities of recall...', end=' ', flush=True)

    p_recall = p_recall_over_time_after_learning(
        agent=learner,
        t_max=t_max,
        n_item=n_item)

    print('Done.\n')

    return {
        'seen': teacher.seen,
        'p_recall': p_recall,
        'questions': questions,
        'replies': replies,
        'successes': successes
    }


def run(student_model, teacher_model,
        student_param=None, n_item=25, grades=(1, ), t_max=500,
        verbose=False,
        force=False):

    """
        :param verbose: Display more stuff
        :param force: Force the computation
        :param teacher_model: Can be one of those:
            * NirajTeacher
            * RandomTeacher
            * AvyaTeacher
            * TugceTeacher
            * RandomTeacher
            * LeitnerTeacher
        :param grades: Levels of difficulty of the kanji selected (1: easiest)
        :param n_item: Positive integer
        (above the number of possible answers displayed)

        :param t_max: Positive integer (zero excluded).
        If grade=1, should be between 6 and 79.

        :param student_model: Class to use for creating the learner. Can be:
            * ActR
            * ActRMeaning
            * ActRGraphic
            * ActRPlus

        :param student_param: dictionary containing the parameters
        when creating the instance of the learner.

            For Act-R models:
            * d: decay rate
            * tau: recalling threshold
            * s: noise/stochasticity
            * g: weight for graphic connection
            * m: weight for semantic connection

            For RL models:
            * alpha: learning rate
            * tau: exploration-exploitation ratio (softmax temperature)

        :return: None
        """

    if student_param is None:

        if student_model == QLearner:
            student_param = {"alpha": 0.1, "tau": 0.05}

        if student_model == ActR:
            student_param = {"d": 0.5, "tau": 0.01, "s": 0.06}

        elif student_model == ActRMeaning:
            student_param = {"d": 0.5, "tau": 0.01, "s": 0.06,
                             "m": 0.1}

        elif student_model == ActRGraphic:
            student_param = {"d": 0.5, "tau": 0.01, "s": 0.06,
                             "g": 0.1}

        elif student_model == ActRPlus:
            student_param = {"d": 0.5, "tau": 0.01, "s": 0.06,
                             "m": 0.1,
                             "g": 0.1}

    assert student_model in \
        (ActR, ActRMeaning, ActRGraphic, ActRPlus, QLearner), \
        "Student model not recognized."
    assert teacher_model in \
        (AvyaTeacher,
         RandomTeacher, LeitnerTeacher), \
        "Teacher model not recognized."

    extension = f'{teacher_model.__name__}_{student_model.__name__}_' \
        f'{dic2string(student_param)}_ni_{n_item}_grade_{grades}_tmax_{t_max}'

    bkp_file = os.path.join('bkp', 'teacher_comparison', f'{extension}.p')

    r = load(bkp_file)
    if r is not None and not force:
        return r
    else:
        r = _produce_data(student_param=student_param,
                          student_model=student_model,
                          teacher_model=teacher_model,
                          t_max=t_max,
                          grades=grades,
                          n_item=n_item,
                          verbose=verbose)

        dump(r, bkp_file)
        return r


def main():

    n_item = 150
    t_max = 4000
    grades = (1, 2)

    student_param = {"d": 0.5, "tau": 0.01, "s": 0.06,
                     "m": 0.1}

    student_model = ActRMeaning
    teacher_models = (RandomTeacher, AvyaTeacher)

    font_size = 8
    label_size = 6
    line_width = 1

    n_rows, n_cols = 5, 2

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 14))

    j = 0

    for teacher_model in teacher_models:
        r = run(student_model=student_model, teacher_model=teacher_model,
                n_item=n_item, t_max=t_max,
                grades=grades, verbose=True)

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

    extension = f'{teacher_models[0].__name__}_' \
        f'{teacher_models[0].__name__}_' \
        f'{student_model.__name__}_' \
        f'{dic2string(student_param)}_ni_{n_item}_grade_{grades}_tmax_{t_max}'
    save_fig(f'teacher_comparison_{extension}.pdf')


if __name__ == "__main__":

    main()
