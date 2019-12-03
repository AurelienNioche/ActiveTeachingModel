import os

from learner.act_r_custom import ActRMeaning
from teacher.leitner import Leitner

import plot.simulation

from simulation.fake import generate_fake_task_param
from simulation.run import run

from utils.string import dic2string

import numpy as np

SCRIPT_NAME = os.path.basename(__file__).split(".")[0]


def main(verbose=False, force=False):

    n_iteration = 1000
    n_item = 200

    student_model = ActRMeaning
    student_param = {"d": 0.01, "tau": 0.01, "s": 0.06, "m": 0.1}

    teacher_model = Leitner
    teacher_param = {'n_item': n_item}

    seed = 123

    np.random.seed(seed)

    task_param = generate_fake_task_param(n_item)

    r = run(
            student_model=student_model,
            student_param=student_param,
            teacher_model=teacher_model,
            teacher_param=teacher_param,
            task_param=task_param,
            n_item=n_item,
            n_iteration=n_iteration,
            verbose=verbose,
            force=force,
        )

    extension = \
        f'{teacher_model.__name__}_{student_model.__name__}_' \
        f'{dic2string(student_param)}_' \
        f'n_item_{n_item}_n_iteration_{n_iteration}_' \
        f'seed_{seed}'

    plot.simulation.summary(
        p_recall=r['p_recall'],
        seen=r['seen'],
        successes=r['successes'],
        extension=extension,
        sub_folder=SCRIPT_NAME
    )


if __name__ == "__main__":

    # for tm in (TraditionalLeitnerTeacher,
    #            LeitnerTeacher,
    #            RandomTeacher,
    #            AvyaTeacher):
    #     main(teacher_model=tm, n_iteration=1000, n_item=30,
    #          normalize_similarity=True)
    main(force=True)
