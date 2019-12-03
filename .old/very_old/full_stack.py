import os
from tqdm import tqdm

from learner.act_r_custom import ActRMeaning
from teacher.active import Active
from fit.pygpgo.classic import BayesianPYGPGOFit
from fit.pygpgo.objective import objective

import multiprocessing as mp
import numpy as np

from simulation.data import Data
from simulation.memory import p_recall_over_time_after_learning
import plot.simulation

from utils.string import dic2string
from utils.backup import dump, load

import argparse


def _run(
        teacher_model,
        n_iteration, grades, n_item,
        normalize_similarity, student_model,
        student_param,
        init_eval,
        time_out,
        verbose):

    queue_in = mp.Queue()
    queue_out = mp.Queue()

    teacher = teacher_model(n_iteration=n_iteration, n_item=n_item,
                            normalize_similarity=normalize_similarity,
                            grades=grades,
                            verbose=False,
                            # learnt_threshold=0.975,
                            )

    learner = student_model(param=student_param, tk=teacher.tk)

    iterator = tqdm(range(n_iteration)) if not verbose else range(n_iteration)

    model_learner = student_model(
        tk=teacher.tk,
        param=student_model.generate_random_parameters()
    )

    # f = BayesianPYGPGOTimeoutFit(verbose=False)
    f = BayesianPYGPGOFit(verbose=False, n_jobs=4, max_evals=3)
    # seen = None
    # learnt = None
    # learnt_model = None
    # which_learnt = None
    # which_learnt_model = Non

    # import copy

    # model_learner.set_parameters(learner.param)

    best_value, obj_value = None, None

    for t in iterator:

        question, possible_replies = teacher.ask(
            agent=model_learner,
            make_learn=False)

        # reply = learner.decide(
        #     question=question,
        #     possible_replies=possible_replies)

        p_r = learner.p_recall(item=question)

        if p_r > np.random.random():
            reply = question
        else:
            reply = -1

        teacher.register_question_and_reply(question=question, reply=reply,
                                            possible_replies=possible_replies)

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

                new_p_recall = p_recall[teacher.taboo]
                new_p_recall_model = p_recall_model[teacher.taboo]
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

        best_param = f.evaluate(
            data=data_view,
            model=student_model,
            tk=teacher.tk,
            init_evals=init_eval,
            max_iter=10,
        )
            # time_out=time_out,
            # queue_in=queue_in, queue_out=queue_out)
        model_learner.set_cognitive_parameters(best_param.copy())

        obj_value = objective(
            data=data_view,
            model=student_model,
            tk=teacher.tk,
            param=student_param,
            show=False
        )

        best_value = objective(
            data=data_view,
            model=student_model,
            tk=teacher.tk,
            param=best_param,
            show=False
        )

        learner.learn(item=question)
        model_learner.learn(item=question)

    queue_in.put('stop')

    p_recall_hist = p_recall_over_time_after_learning(
        agent=learner,
        n_iteration=n_iteration,
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


def main(student_model=None, teacher_model=None,
         student_param=None,
         n_item=30, grades=(1, ), n_iteration=1000,
         normalize_similarity=True, force=False, plot_fig=True,
         init_eval=3, verbose=True,
         time_out=10,
         ):

    if student_model is None:
        student_model = ActRMeaning

    if student_param is None:
        student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02}

    if teacher_model is None:
        teacher_model = Active

    extension = \
        f'full_stack_' \
        f'{teacher_model.__name__}_{student_model.__name__}_' \
        f'{dic2string(student_param)}_' \
        f'ni_{n_item}_grade_{grades}_tmax_{n_iteration}_' \
        f'norm_{normalize_similarity}_' \
        f'time_out_{time_out}'

    bkp_file = os.path.join('../bkp', 'full_stack', f'{extension}.p')

    r = load(bkp_file)
    if r is None or force:
        r = _run(
            student_model=student_model,
            teacher_model=teacher_model,
            student_param=student_param,
            n_item=n_item,
            grades=grades,
            n_iteration=n_iteration,
            normalize_similarity=normalize_similarity,
            init_eval=init_eval,
            time_out=time_out,
            verbose=verbose
        )

        dump(r, bkp_file)

    if plot_fig:

        plot.simulation.summary(
            p_recall=r['p_recall'],
            seen=r['seen'],
            successes=r['successes'],
            extension=extension)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no_fig', '-n', action='store_true', default=False,
                        dest='no_fig',
                        help='Do not create fig')

    parser.add_argument('--force', '-f', default=True, action='store_true',
                        dest='force',
                        help='Force the execution')

    args = parser.parse_args()
    main(plot_fig=not args.no_fig, force=args.force)
