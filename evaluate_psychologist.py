import multiprocessing
import os
import pickle

import numpy as np
from tqdm import tqdm

from fit.scipy import Minimize
from fit.pygpgo import PyGPGO
from fit.gpyopt import Gpyopt

from learner.act_r_custom import ActRMeaning
from learner.act_r import ActR
from learner.rl import QLearner
from learner.half_life import HalfLife
# from learner.simplified import ActROneParam, ActRTwoParam

from teacher.leitner import Leitner
from teacher.random import RandomTeacher

from psychologist.psychologist import SimplePsychologist

from simulation.fake import generate_fake_task_param

import plot.evaluate_psychologist

from utils.utils import dic2string


SCRIPT_NAME = os.path.basename(__file__).split(".")[0]

N_POSSIBLE_REPLIES = 6

DATA_FOLDER = os.path.join("bkp", SCRIPT_NAME)

os.makedirs(DATA_FOLDER, exist_ok=True)


def run(
        teacher_model,
        student_model,
        student_param,
        task_param,
        fit_class,
        n_iteration,
        n_item):

    hist_item = []
    hist_success = []
    # seen = np.zeros((n_item, n_iteration), dtype=bool)

    hist_best_param = []

    teacher = teacher_model(verbose=False, n_item=n_item)

    learner = student_model(
        param=student_param,
        n_iteration=n_iteration,
        **task_param)

    iterator = tqdm(range(n_iteration))

    for t in iterator:

        item = teacher.ask(
            t=t,
            n_item=n_item,
            n_iteration=n_iteration,
            hist_item=hist_item,
            hist_success=hist_success,
            task_param=task_param,
            student_param=student_param,
            student_model=student_model)

        recall = learner.recall(item=item)

        hist_item.append(item)
        hist_success.append(recall)

        # seen[item, t:] = 1

        learner.learn(item=item)

        f = fit_class(model=student_model)
        f.evaluate(
            hist_question=hist_item,
            hist_success=hist_success,
            task_param=task_param
        )

        best_param = f.best_param
        if 'known_param' in task_param:
            best_param.update(task_param['known_param'])
        hist_best_param.append(best_param)

    return hist_best_param


def main():

    fit_class = Gpyopt
    n_iteration = 500
    n_item = 200
    seed = 123
    force = False

    np.random.seed(seed)

    # student_model = ActROneParam
    # student_param = {'d': 0.4743865506892237, 'tau': -12.99017100266079,
    #                  's': 2.2453482763450543}
    # student_model.generate_random_parameters()

    student_model = HalfLife
    known_param = {
        "beta": 0.02
    }

    unknown_param = {
        "alpha": 0.2
    }

    student_param = {**known_param, **unknown_param}
    task_param = generate_fake_task_param(n_item=n_item)

    task_param['known_param'] = known_param
    task_param['unknown_param'] = unknown_param

    full_data = {}

    for teacher_model in (SimplePsychologist, RandomTeacher, Leitner):

        extension = \
            f'{student_model.__name__}{student_model.version}_' \
            f'{teacher_model.__name__}{teacher_model.version}_' \
            f'{fit_class.__name__}_' \
            f'unknown_param_{dic2string(unknown_param)}_' \
            f'known_param_{dic2string(known_param)}_' \
            f'n_item_{n_item}_' \
            f'n_iteration_{n_iteration}_' \
            f'seed_{seed}'

        file_path = os.path.join(
            DATA_FOLDER,
            f"{extension}.p")

        if not os.path.exists(file_path) or force:

            data = run(
                task_param=task_param,
                teacher_model=teacher_model,
                student_model=student_model,
                student_param=student_param,
                fit_class=fit_class,
                n_iteration=n_iteration, n_item=n_item)

            pickle.dump(data, open(file_path, 'wb'))

        else:
            data = pickle.load(open(file_path, 'rb'))

        full_data[teacher_model.__name__] = data

        # print(teacher_model.__name__)
        # print(data)
        # print()

    plot.evaluate_psychologist.plot(
        data=full_data,
        true_parameters=student_param,
        extension=extension)


if __name__ == "__main__":

    main()
