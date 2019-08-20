import os

from learner.act_r_custom import ActRMeaning
from teacher.random import RandomTeacher
from teacher.leitner import Leitner
from teacher.active import Active

import plot.simulation

from simulation.compute import p_recall_over_time_after_learning
from simulation.fake import generate_fake_task_param

from utils.utils import dic2string, load, dump

import numpy as np

from tqdm import tqdm


class Run:

    learnt_threshold = 0.95

    def __init__(
            self,
            n_item,
            n_iteration,
            task_param,
            student_model,
            student_param,
            teacher_model,
            teacher_param,
            verbose=False,
    ):

        self.n_iteration = n_iteration
        self.n_item = n_item
        self.task_param = task_param
        self.student_model = student_model
        self.student_param = student_param
        self.verbose = verbose

        self.hist_item = np.full(n_iteration, -99)
        self.hist_success = np.zeros(n_iteration, dtype=bool)
        self.seen = np.zeros((n_item, n_iteration), dtype=bool)

        self.teacher = teacher_model(verbose=False, **teacher_param)

        self.learner = student_model(
            param=student_param,
            n_iteration=n_iteration,
            **task_param)

    def run(self):

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

        p_recall_hist = p_recall_over_time_after_learning(
            agent=self.learner,
            n_iteration=self.n_iteration,
            n_item=self.n_item,
        )

        return {
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


# def _run(
#         teacher_model,
#         teacher_param,
#         student_model,
#         student_param,
#         verbose):
#
#     teacher = teacher_model(
#         handle_similarities=True,
#         verbose=verbose,
#         **teacher_param
#     )
#
#     learner = student_model(
#         param=student_param,
#         tk=teacher.tk)
#
#     questions, replies, successes = teacher.teach(agent=learner)
#
#     seen = teacher.seen
#
#     # Compute the probability of recall over time
#     p_recall = p_recall_over_time_after_learning(
#         agent=learner,
#         n_iteration=teacher.tk.t_max,
#         n_item=teacher.tk.n_item)
#
#     return {
#         'p_recall': p_recall,
#         'seen': seen,
#         'successes': successes
#     }


def main(verbose=False, force=False):

    n_iteration = 1000
    n_item = 30

    student_model = ActRMeaning
    student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02}

    teacher_model = Leitner
    teacher_param = {'n_item': n_item}

    seed = 123

    np.random.seed(seed)

    task_param = generate_fake_task_param(n_item)

    extension = \
        f'{os.path.basename(__file__).split(".")[0]}_' \
        f'{teacher_model.__name__}_{student_model.__name__}_' \
        f'{dic2string(student_param)}_' \
        f'ni_{n_item}_n_iteration_{n_iteration}_' \
        f'seed_{seed}'

    bkp_file = os.path.join('bkp', 'simulation', f'{extension}.p')

    r = load(bkp_file)
    if r is None or force:
        run = Run(
            student_model=student_model,
            student_param=student_param,
            teacher_model=teacher_model,
            teacher_param=teacher_param,
            task_param=task_param,
            n_item=n_item,
            n_iteration=n_iteration,
            verbose=verbose
        )
        r = run.run()

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
    main(force=True)
