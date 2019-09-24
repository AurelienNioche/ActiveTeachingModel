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
        student_model,
        fit_class,
        n_iteration=300,
        n_item=30,
        seed=123):

    np.random.seed(seed)

    student_param = student_model.generate_random_parameters()

    print("true param", student_param)
    task_param = generate_fake_task_param(n_item=n_item)

    hist_item = np.full(n_iteration, -99)
    hist_success = np.zeros(n_iteration, dtype=bool)
    seen = np.zeros((n_item, n_iteration), dtype=bool)

    teacher = Psychologist(verbose=False, fit_class=fit_class)

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

        hist_item[t] = item
        hist_success[t] = recall

        seen[item, t:] = 1

        learner.learn(item=item)

    return teacher.hist_best_param


def main(student_model, teacher_model, fit_class, n_sim=10,
         n_iteration=1000, n_item=30, seed=123,
         force=False):

    extension = \
        f'_{student_model.__name__}{student_model.version}_' \
        f'_{teacher_model.__name__}_' \
        f'_{fit_class.__name__}_' \
        f'n_sim_{n_sim}_' \
        f'n_item_{n_item}_' \
        f'n_iteration_{n_iteration}' \

    file_path = os.path.join(
        DATA_FOLDER,
        f"{extension}.p")

    if not os.path.exists(file_path) or force:

        data = run(
            student_model=student_model,
            teacher_model=teacher_model,
            fit_class=fit_class,
            n_iteration=n_iteration, n_item=n_item, seed=seed)

        pickle.dump(data, open(file_path, 'wb'))

    else:
        data = pickle.load(open(file_path, 'rb'))

    plot.evaluate_psychologist.plot(
        data=data,
        extension=extension)


if __name__ == "__main__":

    main(student_model=ActR,
         fit_class=Gpyopt,
         n_sim=1,
         n_item=300,
         n_iteration=500,
         force=True)
