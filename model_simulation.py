import os

from learner.act_r_custom import ActRMeaning
from teacher.random import RandomTeacher
from teacher.leitner import LeitnerTeacher
from teacher.complete_leitner import TraditionalLeitnerTeacher
from teacher.avya import AvyaTeacher

import plot.simulation

from simulation.memory import p_recall_over_time_after_learning

from utils.utils import dic2string, load, dump


def _run(
        teacher_model,
        t_max, grades, n_item, normalize_similarity,
        student_model,
        student_param,
        verbose):

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

    return {
        'p_recall': p_recall,
        'seen': seen,
        'successes': successes
    }


def main(t_max=300, n_item=30, grades=(1, ),
         student_model=None,
         student_param=None,
         teacher_model=None, verbose=False,
         normalize_similarity=True,
         force=False):

    if student_model is None:
        student_model = ActRMeaning

    if student_param is None:
        student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02}

    if teacher_model is None:
        teacher_model = RandomTeacher

    extension = f'{teacher_model.__name__}_{student_model.__name__}_' \
                f'{dic2string(student_param)}_' \
                f'ni_{n_item}_grade_{grades}_tmax_{t_max}_' \
                f'norm_{normalize_similarity}'

    bkp_file = os.path.join('bkp', 'simulation', f'{extension}.p')

    r = load(bkp_file)
    if r is None or force:
        r = _run(
            student_model=student_model,
            teacher_model=teacher_model,
            student_param=student_param,
            n_item=n_item,
            grades=grades,
            t_max=t_max,
            normalize_similarity=normalize_similarity,
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
        teacher_model=AvyaTeacher,
        t_max=1000, n_item=30, normalize_similarity=True,
        force=True
    )
