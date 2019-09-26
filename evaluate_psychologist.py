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

from teacher.leitner import Leitner
from teacher.random import RandomTeacher

from psychologist.psychologist import Psychologist

from simulation.fake import generate_fake_task_param

import plot.evaluate_psychologist


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

    if teacher_model == Psychologist:
        teacher = Psychologist(
            verbose=False, n_item=n_item,
            fit_class=fit_class)

    else:
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

        if teacher_model != Psychologist:
            f = fit_class(model=student_model)
            f.evaluate(
                hist_question=hist_item,
                hist_success=hist_success,
                task_param=task_param
            )
            hist_best_param.append(f.best_param)

        else:
            teacher.update_estimates(
                student_model=student_model,
                task_param=task_param,
                hist_item=hist_item,
                hist_success=hist_success,
                t=t
            )

    if teacher_model == Psychologist:
        return teacher.hist_best_param
    else:
        return hist_best_param


def main():

    student_model = ActR
    fit_class = Gpyopt
    n_iteration = 250
    n_item = 200
    seed = 123
    force = True

    np.random.seed(seed)

    student_param = {'d': 0.4743865506892237, 'tau': -12.99017100266079,
                     's': 2.2453482763450543}
    # student_model.generate_random_parameters()
    task_param = generate_fake_task_param(n_item=n_item)

    full_data = {}

    for teacher_model in (Psychologist, RandomTeacher, Leitner):

        extension = \
            f'{student_model.__name__}{student_model.version}_' \
            f'_{teacher_model.__name__}_' \
            f'_{fit_class.__name__}_' \
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
