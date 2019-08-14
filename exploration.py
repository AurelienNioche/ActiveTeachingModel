import os
from tqdm import tqdm

from learner.act_r_custom import ActRMeaning
from teacher.active import Active
from fit.bayesian_pygpgo import BayesianPYGPGOFit, objective
from fit.bayesian_pygpgo_timeout import BayesianPYGPGOTimeoutFit

import multiprocessing as mp
import numpy as np

from simulation.data import Data
from simulation.memory import p_recall_over_time_after_learning
import plot.simulation

from utils.utils import dic2string, dump, load

import argparse

from psychologist.psychologist import Psychologist

def _run(
        teacher_model,
        t_max, grades, n_item,
        normalize_similarity, student_model,
        student_param,
        init_eval,
        max_iter,
        verbose):

    # queue_in = mp.Queue()
    # queue_out = mp.Queue()

    teacher = teacher_model(t_max=t_max, n_item=n_item,
                            normalize_similarity=normalize_similarity,
                            grades=grades,
                            verbose=False,
                            # learnt_threshold=0.975,
                            )

    learner = student_model(param=student_param, tk=teacher.tk)

    iterator = tqdm(range(t_max)) if not verbose else range(t_max)

    model_learner = student_model(
        tk=teacher.tk,
        param=student_model.generate_random_parameters()
    )

    # f = BayesianPYGPGOTimeoutFit(verbose=False)
    f = BayesianPYGPGOFit(verbose=False, n_jobs=4)
    # seen = None
    # learnt = None
    # learnt_model = None
    # which_learnt = None
    # which_learnt_model = Non

    # import copy

    # model_learner.set_parameters(learner.param)

    best_value, obj_value = None, None

    param_set = None

    for t in iterator:
        if t > 0:
            r = np.random.random()
            if r < 0.10:
                print("EXPLO")
                question = Psychologist.most_informative(
                    tk=teacher.tk,
                    student_model=student_model,
                    param_set=param_set,
                    questions=teacher.questions,
                    t_max=t+1
                )
            else:
                question, possible_replies = teacher.ask(
                    agent=model_learner,
                    make_learn=False)

        else:
            question = np.random.randint(teacher.tk.n_item)

        p_r = learner.p_recall(item=question)

        if p_r > np.random.random():
            reply = question
        else:
            reply = -1

        teacher.register_question_and_reply(question=question, reply=reply)

        if verbose:

            p_recall = np.zeros(n_item)
            p_recall_model = np.zeros(n_item)
            for i in range(n_item):
                p_recall[i] = learner.p_recall(i)
                p_recall_model[i] = model_learner.p_recall(i)

            learnt = np.sum(p_recall >= 0.95)
            which_learnt = np.where(p_recall >= 0.95)[0]

            learnt_model = np.sum(p_recall_model >= 0.95)
            which_learnt_model = np.where(p_recall_model >= 0.95)[0]

            if t > 0:

                print()
                discrepancy = 'Param disc.'
                for k in sorted(list(model_learner.param.keys())):
                    discrepancy += \
                        f'{k}: ' \
                        f'{model_learner.param[k] - student_param[k]:.3f}; '

                print(discrepancy, '\n')
                print(f"Obj value: {obj_value:.2f}; "
                      f"Best value: {best_value:.2f}")
                print()

                new_p_recall = p_recall[teacher.questions[t-1]]
                new_p_recall_model = p_recall_model[teacher.questions[t-1]]
                print(
                    f'New p recall: {new_p_recall:.3f}; '
                    f'New p recall model: {new_p_recall_model:.3f}')

                print()

                seen = sum(teacher.seen[:, t-1])
                print(f'N seen: {seen}')
                print(f'Learnt: {learnt} {list(which_learnt)}; '
                      f'Learnt model: {learnt_model} '
                      f'{list(which_learnt_model)}')

            print(f'\n ----- T{t} -----')

            success = question == reply
            # np.sum(teacher.learning_progress == teacher.represent_learnt)
            # print(f'Rule: {teacher.rule}')
            print(f'Question: {question}; Success: {int(success)}')
            print()
            print(
                f'P recall: {p_recall[question]:.3}; '
                f'P recall model: {p_recall_model[question]:.3f}')

        data_view = Data(n_item=n_item,
                         questions=teacher.questions[:t + 1],
                         replies=teacher.replies[:t + 1],
                         possible_replies=teacher.possible_replies[:t + 1, :])

        f.evaluate(
            data=data_view,
            model=student_model,
            tk=teacher.tk,
            init_evals=init_eval,
            max_iter=max_iter
        )

        model_learner.set_parameters(f.best_param.copy())

        obj_value = objective(
            data=data_view,
            model=student_model,
            tk=teacher.tk,
            param=student_param,
            show=False
        )

        best_value = f.best_value
        #     objective(
        #     data=data_view,
        #     model=student_model,
        #     tk=teacher.tk,
        #     param=best_param,
        #     show=False
        # )
        param_set = f.history_eval_param

        learner.learn(question=question)
        model_learner.learn(question=question)

    # queue_in.put('stop')

    p_recall_hist = p_recall_over_time_after_learning(
        agent=learner,
        t_max=t_max,
        n_item=n_item,
    )

    return {
        'seen': teacher.seen,
        'p_recall': p_recall_hist,
        'questions': teacher.questions,
        'replies': teacher.replies,
        'successes': teacher.successes,
        'history_best_fit_param': f.history_best_fit_param,
        'history_best_fit_value': f.history_best_fit_value,
    }


def main(force=False):

    student_model = ActRMeaning
    student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02}
    teacher_model = Active
    n_item = 30
    grades = 1,
    t_max = 1000
    normalize_similarity = True
    init_eval = 3
    verbose = True,
    max_iter = 10

    extension = \
        f'{os.path.basename(__file__)}_' \
        f'{teacher_model.__name__}_{student_model.__name__}_' \
        f'{dic2string(student_param)}_' \
        f'ni_{n_item}_grade_{grades}_tmax_{t_max}_' \
        f'norm_{normalize_similarity}_' \
        f'init_eval_{init_eval}_' \
        f'max_iter_{max_iter}'

    bkp_file = os.path.join('bkp',
                            f'{os.path.basename(__file__)}',
                            f'{extension}.p')

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
            init_eval=init_eval,
            max_iter=max_iter,
            verbose=verbose
        )

        dump(r, bkp_file)

    plot.simulation.summary(
        p_recall=r['p_recall'],
        seen=r['seen'],
        successes=r['successes'],
        extension=extension)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--force', '-f',
                        default=False,
                        action='store_true',
                        dest='force',
                        help='Force the execution')

    args = parser.parse_args()

    main(force=args.force)
