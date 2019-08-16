import os

from learner.act_r_custom import ActRMeaning
from teacher.random import RandomTeacher
from teacher.leitner import LeitnerTeacher
from teacher.complete_leitner import TraditionalLeitnerTeacher
from teacher.active import Active, ForceLearning, ActivePlus

import plot.simulation

from simulation.memory import p_recall_over_time_after_learning

from utils.utils import dic2string, load, dump


def _run(
        teacher_model,
        teacher_param,
        student_model,
        student_param,
        verbose):

    teacher = teacher_model(
        handle_similarities=True,
        verbose=verbose,
        **teacher_param
    )

    learner = student_model(
        param=student_param,
        tk=teacher.tk)

    questions, replies, successes = teacher.teach(agent=learner)

    seen = teacher.seen

    # Compute the probability of recall over time
    p_recall = p_recall_over_time_after_learning(
        agent=learner,
        t_max=teacher.tk.t_max,
        n_item=teacher.tk.n_item)

    return {
        'p_recall': p_recall,
        'seen': seen,
        'successes': successes
    }


def main(teacher_param=None,
         student_model=None,
         student_param=None,
         teacher_model=None,
         verbose=False,
         force=False):

    if student_model is None:
        student_model = ActRMeaning

    if student_param is None:
        student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02}

    if teacher_model is None:
        teacher_model = RandomTeacher

    if teacher_param is None:
        teacher_param = {"t_max": 1000, "n_item": 30,
                         "normalize_similarity": True,
                         "grades": (1,)}

    extension = f'{teacher_model.__name__}{teacher_model.version}_' \
                f'{dic2string(teacher_param)}_' \
                f'{student_model.__name__}_' \
                f'{dic2string(student_param)}'

    bkp_file = os.path.join('bkp', 'simulation', f'{extension}.p')

    r = load(bkp_file)
    if r is None or force:
        r = _run(
            teacher_model=teacher_model,
            teacher_param=teacher_param,
            student_model=student_model,
            student_param=student_param,
            verbose=verbose
        )

        dump(r, bkp_file)

    plot.simulation.summary(
        p_recall=r['p_recall'],
        seen=r['seen'],
        successes=r['successes'],
        extension=extension
    )


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
        teacher_model=ActivePlus,
        teacher_param={"t_max": 1000, "n_item": 30,
                       "normalize_similarity": True,
                       "grades": (1, ),
                       "depth": 2},
        force=True
    )
