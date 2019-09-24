import os

import numpy as np
from tqdm import tqdm

from learner.act_r import ActR
from learner.act_r_custom import ActRMeaning, ActRGraphic
from learner.rl import QLearner

from teacher.active import Active
from teacher.leitner import Leitner
from teacher.random import RandomTeacher

from psychologist.psychologist import Psychologist

from utils.utils import dic2string, load, dump

from . p_recall import p_recall_over_time_after_learning
from . fake import generate_fake_task_param


class Run:

    learnt_threshold = 0.95

    def __init__(
            self,
            n_item,
            n_iteration,
            student_model,
            student_param,
            teacher_model,
            verbose=False,
            task_param=None,
            teacher_param=None,
    ):

        self.n_iteration = n_iteration
        self.n_item = n_item

        self.student_model = student_model
        self.student_param = student_param
        self.verbose = verbose

        self.task_param = task_param
        self.teacher_param = teacher_param

        self.hist_item = np.full(n_iteration, -99)
        self.hist_success = np.zeros(n_iteration, dtype=bool)
        self.seen = np.zeros((n_item, n_iteration), dtype=bool)

        self.teacher = teacher_model(verbose=False,
                                     **self.teacher_param)

        self.learner = student_model(
            param=student_param,
            n_iteration=n_iteration,
            **self.task_param)

        # Will stock the results
        self.results = None

    def run(self, compute_p_recall_hist=True):

        iterator = tqdm(range(self.n_iteration)) \
            if not self.verbose else range(self.n_iteration)

        for t in iterator:

            item = self.teacher.ask(
                t=t,
                n_item=self.n_item,
                n_iteration=self.n_iteration,
                hist_item=self.hist_item,
                hist_success=self.hist_success,
                task_param=self.task_param,
                student_param=self.student_param,
                student_model=self.student_model)

            recall = self.learner.recall(item=item)

            self.hist_item[t] = item
            self.hist_success[t] = recall

            self.seen[item, t:] = 1

            if self.verbose:
                self.print(t)

            self.learner.learn(item=item)

        if compute_p_recall_hist:
            p_recall_hist = p_recall_over_time_after_learning(
                agent=self.learner,
                n_iteration=self.n_iteration,
                n_item=self.n_item,
            )
        else:
            p_recall_hist = None

        self.results = {
            'seen': self.seen,
            'p_recall': p_recall_hist,
            'questions': self.hist_item,
            'successes': self.hist_success,
        }

    def print(self, t):

        p_recall = np.zeros(self.n_item)
        for i in range(self.n_item):
            p_recall[i] = self.learner.p_recall(i)

        learnt = \
            np.sum(p_recall >= self.learnt_threshold)
        which_learnt = \
            np.where(p_recall >= self.learnt_threshold)[0]

        if t > 0:
            new_p_recall = p_recall[self.hist_item[t - 1]]
            print(
                f'New p recall: {new_p_recall:.3f}')

            print()

            seen = sum(self.seen[:, t - 1])
            print(f'N seen: {seen}')
            print(f'Learnt: {learnt} {list(which_learnt)}')

        print(f'\n ----- T{t} -----')

        success = self.hist_success[t]
        print(f'Question: {self.hist_item[t]}; '
              f'Success: {int(success)}')
        print()
        print(
            f'P recall: {p_recall[self.hist_item[t]]:.3}')


def run(student_model, teacher_model,
        student_param=None,
        teacher_param=None,
        task_param=None,
        n_item=25,
        n_iteration=500,
        compute_p_recall_hist=True,
        verbose=False,
        force=False):

    """
        :param compute_p_recall_hist: bool
        :param task_param: dictionary containing the parameters of the task
        (i.e., the semantic connections)
        :param teacher_param: dictionary containing the parameters when
        creating the instance of the teacher,
        :param verbose: Display more stuff
        :param force: Force the computation
        :param teacher_model: Can be one of those:
            * RandomTeacher
            * Active
            * Leitner
        :param n_item: Positive integer
        (it has to be superior to the number of possible answers displayed)

        :param n_iteration: Positive integer (zero excluded).

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

        if student_model == ActR:
            student_param = {"d": 0.5, "tau": 0.01, "s": 0.06}

        elif student_model == ActRMeaning:
            student_param = {"d": 0.5, "tau": 0.01, "s": 0.06,
                             "m": 0.1}

        elif student_model == ActRGraphic:
            student_param = {"d": 0.5, "tau": 0.01, "s": 0.06,
                             "g": 0.1}

        # elif student_model == ActRPlus:
        #     student_param = {"d": 0.5, "tau": 0.01, "s": 0.06,
        #                      "m": 0.1,
        #                      "g": 0.1}
        elif student_model == QLearner:
            student_param = {
                "alpha": 0.1,
                "tau": 0.05
            }

    if task_param is None:
        task_param = generate_fake_task_param(n_item=n_item)

    if teacher_param is None:
        teacher_param = {'n_item': n_item}

    assert student_model in \
        (ActR, ActRMeaning, ActRGraphic, QLearner), \
        "Student model not recognized."
    assert teacher_model in \
        (Active, RandomTeacher, Leitner, Psychologist), \
        "Teacher model not recognized."

    extension = f'{teacher_model.__name__}_{student_model.__name__}_' \
        f'{dic2string(student_param)}_' \
        f'{dic2string(teacher_param)}_' \
        f'n_item_{n_item}_n_iteration_{n_iteration}_' \
        f'p_recall_hist_{compute_p_recall_hist}'

    bkp_file = os.path.join('bkp', 'simulation', f'{extension}.p')

    r = load(bkp_file)
    if r is None or force:
        r = Run(
            student_param=student_param,
            student_model=student_model,
            teacher_model=teacher_model,
            teacher_param=teacher_param,
            task_param=task_param,
            n_iteration=n_iteration,
            n_item=n_item,
            verbose=verbose)
        r.run(compute_p_recall_hist=compute_p_recall_hist)

        dump(r, bkp_file)

    return r.results
