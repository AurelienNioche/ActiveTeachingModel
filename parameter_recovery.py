import multiprocessing
import os
import pickle

import numpy as np
from tqdm import tqdm

from fit.scipy import Minimize
from fit.pygpgo import PyGPGO

from learner.act_r_custom import ActRMeaning
from learner.act_r import ActR
from learner.rl import QLearner

from teacher.leitner import Leitner
from teacher.random import RandomTeacher

from psychologist.psychologist import Psychologist

from simulation.run import run
from simulation.fake import generate_fake_task_param

import plot.parameter_recovery


SCRIPT_NAME = os.path.basename(__file__).split(".")[0]

N_POSSIBLE_REPLIES = 6

DATA_FOLDER = os.path.join("bkp", SCRIPT_NAME)

os.makedirs(DATA_FOLDER, exist_ok=True)


class SimulationAndFit:

    def __init__(self,
                 student_model,
                 teacher_model,
                 fit_class, n_iteration=300, n_item=30,
                 verbose=False):

        self.student_model = student_model
        self.teacher_model = teacher_model

        self.n_item = n_item
        self.n_iteration = n_iteration

        self.fit_class = fit_class

        self.verbose = verbose

    def __call__(self, seed):

        np.random.seed(seed)

        param = self.student_model.generate_random_parameters()

        print("true param", param)
        task_param = generate_fake_task_param(n_item=self.n_item)

        r = run(
            student_param=param,
            student_model=self.student_model,
            teacher_model=self.teacher_model,
            n_item=self.n_item,
            n_iteration=self.n_iteration,
            task_param=task_param,
            compute_p_recall_hist=False
        )

        f = self.fit_class(model=self.student_model)

        rf = f.evaluate(
            hist_question=r["questions"],
            hist_success=r["successes"],
            task_param=task_param)

        return \
            {
                "initial": param,
                "recovered": rf["best_param"],
            }


def main(student_model, teacher_model, fit_class, n_sim=10,
         n_iteration=1000, n_item=30,
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

        seeds = range(n_sim)

        pool = multiprocessing.Pool()
        results = list(tqdm(pool.imap_unordered(
            SimulationAndFit(student_model=student_model,
                             teacher_model=teacher_model,
                             fit_class=fit_class,
                             n_iteration=n_iteration, n_item=n_item), seeds
        ), total=n_sim))

        r_keys = list(results[0].keys())
        param_names = results[0][r_keys[0]].keys()

        data = {
            pn:
                {
                    k: [] for k in r_keys
                }
            for pn in param_names
        }

        for r in results:
            for k in r_keys:
                for pn in param_names:
                    data[pn][k].append(r[k][pn])

        pickle.dump(data, open(file_path, 'wb'))

    else:
        data = pickle.load(open(file_path, 'rb'))

    plot.parameter_recovery.plot(
        data=data,
        extension=extension)


if __name__ == "__main__":

    main(student_model=ActR,
         teacher_model=Psychologist,
         fit_class=PyGPGO,
         n_sim=1,
         n_item=300,
         n_iteration=500,
         force=True)
