import os
from tqdm import tqdm

from learner.act_r_custom import ActRMeaning
from teacher.active import Active

from fit.pygpgo.objective import objective

import numpy as np

from simulation.compute import p_recall_over_time_after_learning
from simulation.fake import generate_fake_task_param

import plot.simulation

from utils.utils import dic2string, dump, load

import argparse

import multiprocessing as mp

from psychologist.psychologist import Psychologist


class Run:

    def __init__(
            self,
            n_item, n_iteration,
            student_model,
            student_param,
            task_param,
            teacher_model,
            init_eval,
            max_iter,
            exploration_ratio,
            testing_period,
            n_jobs=mp.cpu_count(),
            timeout=None,
            verbose=False,
    ):

        self.exploration_ratio = exploration_ratio
        self.testing_period = testing_period
        self.n_iteration = n_iteration
        self.n_item = n_item
        self.task_param = task_param
        self.student_model = student_model
        self.student_param = student_param
        self.timeout = timeout
        self.verbose = verbose

        self.hist_item = np.full(n_iteration, -99)
        self.hist_success = np.zeros(n_iteration, dtype=bool)
        self.seen = np.zeros((n_item, n_iteration), dtype=bool)

        self.best_param = student_model.generate_random_parameters()

        self.teacher = teacher_model(
            n_item=n_item,
            verbose=False)

        self.psychologist = Psychologist(
            student_model=student_model,
            task_param=task_param,
            testing_period=testing_period,
            exploration_ratio=exploration_ratio,
            n_jobs=n_jobs,
            init_eval=init_eval,
            max_iter=max_iter,
            timeout=timeout
        )

        self.learner = student_model(
            param=student_param,
            n_iteration=n_iteration,
            **task_param)

        if self.verbose:
            self.model_learner = student_model(
                n_iteration=n_iteration,
                param=self.best_param,
                **task_param
            )
            self.learnt_threshold = 0.95

        self.best_value, \
        self.obj_value, self.eval_param, self.exploration = \
            None, None, None, None

    def run(self):

        iterator = tqdm(range(self.n_iteration)) \
            if not self.verbose else range(self.n_iteration)

        for t in iterator:

            self.exploration = self.psychologist.is_time_for_exploration(t)

            if self.exploration:
                item = self.psychologist.most_informative(
                    n_item=self.n_item,
                    n_iteration=self.n_iteration,
                    task_param=self.task_param,
                    student_model=self.student_model,
                    eval_param=self.eval_param,
                    hist_item=self.hist_item,
                    t=t
                )

            else:
                item = self.teacher.ask(
                    t=t,
                    n_item=self.n_item,
                    n_iteration=self.n_iteration,
                    hist_item=self.hist_item,
                    hist_success=self.hist_success,
                    task_param=self.task_param,
                    student_param=self.best_param,
                    student_model=self.student_model)

            recall = self.learner.recall(item=item)

            self.hist_item[t] = item
            self.hist_success[t] = recall

            self.seen[item, t:] = 1

            if self.verbose:
                self.print(t)

            self.best_param = self.psychologist.update_estimates(
                hist_success=self.hist_success,
                hist_item=self.hist_item, t=t)

            if self.verbose:
                self.obj_value = objective(
                    hist_success=self.hist_success,
                    hist_item=self.hist_item,
                    model=self.student_model,
                    t=t,
                    param=self.student_param,
                    task_param=self.task_param,
                    show=False
                )

            self.best_value = self.psychologist.opt.best_value
            self.eval_param = self.psychologist.opt.eval_param

            self.learner.learn(item=item)

            if self.verbose:
                self.model_learner.learn(item=item)

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
            'history_best_fit_param': self.psychologist.opt.hist_best_param,
            'history_best_fit_value': self.psychologist.opt.hist_best_value,
        }

    def print(self, t):

        p_recall = np.zeros(self.n_item)
        p_recall_model = np.zeros(self.n_item)
        for i in range(self.n_item):
            p_recall[i] = self.learner.p_recall(i)
            p_recall_model[i] = self.model_learner.p_recall(i)

        learnt = \
            np.sum(p_recall >= self.learnt_threshold)
        which_learnt = \
            np.where(p_recall >= self.learnt_threshold)[0]

        learnt_model = \
            np.sum(p_recall_model >= self.learnt_threshold)
        which_learnt_model = \
            np.where(p_recall_model >= self.learnt_threshold)[0]

        if t > 0:

            print()
            discrepancy = 'Param disc.'
            for k in sorted(list(self.model_learner.param.keys())):
                discrepancy += \
                    f'{k}: ' \
                    f'{self.model_learner.param[k] - self.student_param[k]:.3f}; '

            print(discrepancy, '\n')
            print(f"Obj value: {self.obj_value:.2f}; "
                  f"Best value: {self.best_value:.2f}")
            print()

            new_p_recall = p_recall[self.hist_item[t - 1]]
            new_p_recall_model = p_recall_model[self.hist_item[t - 1]]
            print(
                f'New p recall: {new_p_recall:.3f}; '
                f'New p recall model: {new_p_recall_model:.3f}')

            print()

            seen = sum(self.seen[:, t - 1])
            print(f'N seen: {seen}')
            print(f'Learnt: {learnt} {list(which_learnt)}; '
                  f'Learnt model: {learnt_model} '
                  f'{list(which_learnt_model)}')

        print(f'\n ----- T{t} -----')

        print(f'Exploration: {self.exploration}')

        success = self.hist_success[t]
        # np.sum(teacher.learning_progress == teacher.represent_learnt)
        # print(f'Rule: {teacher.rule}')
        print(f'Question: {self.hist_item[t]}; '
              f'Success: {int(success)}')
        print()
        print(
            f'P recall: {p_recall[self.hist_item[t]]:.3}; '
            f'P recall model: {p_recall_model[self.hist_item[t]]:.3f}')


def main(force=True):

    student_model = ActRMeaning
    student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02}
    teacher_model = Active
    n_item = 30
    n_iteration = 1000
    init_eval = 3
    verbose = True
    max_iter = 100
    timeout = 5
    exploration_ratio = 0.1
    testing_period = 0
    seed = 123

    np.random.seed(seed)

    task_param = generate_fake_task_param(n_item)

    # For fig
    mean_window = 10

    extension = \
        f'{os.path.basename(__file__).split(".")[0]}_' \
        f'{teacher_model.__name__}_{student_model.__name__}_' \
        f'{dic2string(student_param)}_' \
        f'ni_{n_item}_n_iteration_{n_iteration}_' \
        f'init_eval_{init_eval}_' \
        f'max_iter_{max_iter}_' \
        f'time_out_{timeout}_' \
        f'exploration_ratio_{exploration_ratio}_' \
        f'testing_period_{testing_period}_' \
        f'seed_{seed}'

    bkp_file = os.path.join('bkp',
                            f'{os.path.basename(__file__).split(".")[0]}',
                            f'{extension}.p')

    r = load(bkp_file)
    if r is None or force:
        run = Run(
            student_model=student_model,
            teacher_model=teacher_model,
            student_param=student_param,
            task_param=task_param,
            n_item=n_item,
            n_iteration=n_iteration,
            init_eval=init_eval,
            max_iter=max_iter,
            timeout=timeout,
            exploration_ratio=exploration_ratio,
            testing_period=testing_period,
            verbose=verbose
        )
        r = run.run()

        dump(r, bkp_file)

    plot.simulation.summary(
        p_recall=r['p_recall'],
        seen=r['seen'],
        successes=r['successes'],
        extension=extension,
        window=mean_window,
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--force', '-f',
                        default=True,
                        action='store_true',
                        dest='force',
                        help='Force the execution')

    args = parser.parse_args()

    main(force=args.force)
