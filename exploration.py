import os
from tqdm import tqdm

from learner.act_r_custom import ActRMeaning
from teacher.active import Active
from fit.pygpgo.classic import PYGPGOFit
from fit.pygpgo.objective import objective

import numpy as np

from simulation.memory import p_recall_over_time_after_learning
import plot.simulation

from utils.utils import dic2string, dump, load

import argparse

import multiprocessing as mp

from psychologist.psychologist import Psychologist


class Run:

    def __init__(
            self,
            teacher_model,
            grades, n_item, n_iteration,
            normalize_similarity, student_model,
            student_param,
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
        self.student_model = student_model
        self.student_param = student_param
        self.timeout = timeout
        self.verbose = verbose

        self.hist_item = np.zeros(n_iteration, dtype=int)
        self.hist_success = np.zeros(n_iteration, dtype=bool)

        self.teacher = teacher_model(
            t_max=n_iteration, n_item=n_item,
            normalize_similarity=normalize_similarity,
            grades=grades,
            verbose=False)

        self.psychologist = Psychologist(
            student_model=student_model, tk=self.teacher.tk,
            n_jobs=n_jobs, init_eval=init_eval, max_iter=max_iter,
            timeout=timeout
        )

        self.learner = student_model(param=student_param, tk=self.teacher.tk)

        self.best_value, self.obj_value, self.param_set, self.exploration = \
            None, None, None, None

    def run(self):

        np.random.seed(123)

        iterator = tqdm(range(self.n_iteration)) \
            if not self.verbose else range(self.n_iteration)

        for t in iterator:
            self.exploration = self.psychologist.is_time_for_exploration(
                t
            )

            if self.exploration:
                question = self.psychologist.most_informative(
                    tk=self.teacher.tk,
                    student_model=self.student_model,
                    eval_param=self.param_set,
                    questions=self.teacher.questions,
                    t_max=t+1
                )
            else:
                item, possible_replies = self.teacher.ask(
                    agent=self.psychologist.model_learner,
                    make_learn=False)

            recall = self.learner.recall(item=item)

            self.hist_item[t] = item
            self.hist_success[t] = item

            if self.verbose:
                self.print(t)

            self.opt.evaluate(
                data=data_view, model=self.student_model, tk=self.teacher.tk)

            self.model_learner.set_parameters(self.opt.best_param.copy())

            if self.verbose:
                self.obj_value = objective(
                    data=data_view,
                    model=self.student_model,
                    tk=self.teacher.tk,
                    param=self.student_param,
                    show=False
                )

            self.best_value = self.opt.best_value
            self.param_set = self.opt.eval_param

            self.learner.learn(item=question)
            self.model_learner.learn(item=question)

        p_recall_hist = p_recall_over_time_after_learning(
            agent=self.learner,
            t_max=self.teacher.tk.t_max,
            n_item=self.teacher.tk.n_item,
        )

        return {
            'seen': self.teacher.seen,
            'p_recall': p_recall_hist,
            'questions': self.teacher.questions,
            'replies': self.teacher.replies,
            'successes': self.teacher.successes,
            'history_best_fit_param': self.opt.hist_best_param,
            'history_best_fit_value': self.opt.hist_best_value,
        }

    def print(self, t):

        p_recall = np.zeros(self.teacher.tk.n_item)
        p_recall_model = np.zeros(self.teacher.tk.n_item)
        for i in range(self.teacher.tk.n_item):
            p_recall[i] = self.learner.p_recall(i)
            p_recall_model[i] = self.model_learner.p_recall(i)

        learnt = \
            np.sum(p_recall >= self.teacher.learnt_threshold)
        which_learnt = \
            np.where(p_recall >= self.teacher.learnt_threshold)[0]

        learnt_model = \
            np.sum(p_recall_model >= self.teacher.learnt_threshold)
        which_learnt_model = \
            np.where(p_recall_model >= self.teacher.learnt_threshold)[0]

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

            new_p_recall = p_recall[self.teacher.questions[t - 1]]
            new_p_recall_model = p_recall_model[self.teacher.questions[t - 1]]
            print(
                f'New p recall: {new_p_recall:.3f}; '
                f'New p recall model: {new_p_recall_model:.3f}')

            print()

            seen = sum(self.teacher.seen[:, t - 1])
            print(f'N seen: {seen}')
            print(f'Learnt: {learnt} {list(which_learnt)}; '
                  f'Learnt model: {learnt_model} '
                  f'{list(which_learnt_model)}')

        print(f'\n ----- T{t} -----')

        print(f'Exploration: {self.exploration}')

        success = self.teacher.questions[t] == self.teacher.replies[t]
        # np.sum(teacher.learning_progress == teacher.represent_learnt)
        # print(f'Rule: {teacher.rule}')
        print(f'Question: {self.teacher.questions[t]}; '
              f'Success: {int(success)}')
        print()
        print(
            f'P recall: {p_recall[self.teacher.questions[t]]:.3}; '
            f'P recall model: {p_recall_model[self.teacher.questions[t]]:.3f}')


def main(force=False):

    student_model = ActRMeaning
    student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02}
    teacher_model = Active
    n_item = 30
    grades = 1,
    t_max = 1000
    normalize_similarity = True
    init_eval = 3
    verbose = True
    max_iter = 100
    timeout = 5
    exploration_ratio = 0.1
    testing_period = 1000

    # For fig
    mean_window = 10

    extension = \
        f'{os.path.basename(__file__).split(".")[0]}_' \
        f'{teacher_model.__name__}_{student_model.__name__}_' \
        f'{dic2string(student_param)}_' \
        f'ni_{n_item}_grade_{grades}_tmax_{t_max}_' \
        f'norm_{normalize_similarity}_' \
        f'init_eval_{init_eval}_' \
        f'max_iter_{max_iter}_' \
        f'time_out_{timeout}_' \
        f'exploration_ratio_{exploration_ratio}_' \
        f'testing_period_{testing_period}'

    bkp_file = os.path.join('bkp',
                            f'{os.path.basename(__file__).split(".")[0]}',
                            f'{extension}.p')

    r = load(bkp_file)
    if r is None or force:
        run = Run(
            student_model=student_model,
            teacher_model=teacher_model,
            student_param=student_param,
            n_item=n_item,
            grades=grades,
            t_max=t_max,
            normalize_similarity=normalize_similarity,
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
                        default=False,
                        action='store_true',
                        dest='force',
                        help='Force the execution')

    args = parser.parse_args()

    main(force=args.force)
