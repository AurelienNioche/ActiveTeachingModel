import os
from tqdm import tqdm

from learner.act_r_custom import ActRMeaning
from teacher.avya import AvyaTeacher
from fit.bayesian_pygpgo_timeout import BayesianPYGPGOTimeoutFit

import multiprocessing as mp

from simulation.data import Data
from simulation.memory import p_recall_over_time_after_learning
import plot.simulation

from utils.utils import dic2string, dump, load

import argparse


def _run(
        teacher_model,
        t_max, grades, n_item,
        normalize_similarity, student_model,
        student_param, n_cpu,
        init_eval,
        time_out,
        verbose):

    queue_in = mp.Queue()
    queue_out = mp.Queue()

    teacher = teacher_model(t_max=t_max, n_item=n_item,
                            normalize_similarity=normalize_similarity,
                            grades=grades)

    learner = student_model(param=student_param, tk=teacher.tk)

    iterator = tqdm(range(t_max))

    model_learner = student_model(
        tk=teacher.tk,
        param=student_model.generate_random_parameters()
    )

    f = BayesianPYGPGOTimeoutFit(verbose=verbose)

    for t in iterator:

        question, possible_replies = teacher.ask(
            agent=model_learner,
            make_learn=False)

        reply = learner.decide(
            question=question,
            possible_replies=possible_replies)

        learner.learn(question=question)

        teacher.register_question_and_reply(question=question, reply=reply,
                                            possible_replies=possible_replies)

        data_view = Data(n_items=n_item,
                         questions=teacher.questions[:t + 1],
                         replies=teacher.replies[:t + 1],
                         possible_replies=teacher.possible_replies[:t + 1, :])

        f.evaluate(data=data_view,
                   model=student_model,
                   tk=teacher.tk,
                   time_out=time_out, init_evals=init_eval,
                   queue_in=queue_in, queue_out=queue_out)

        model_learner.set_parameters(f.best_param)

        model_learner.learn(question=question)

    p_recall = p_recall_over_time_after_learning(
        agent=learner,
        t_max=t_max,
        n_item=n_item)

    return {
        'seen': teacher.seen,
        'p_recall': p_recall,
        'questions': teacher.questions,
        'replies': teacher.replies,
        'successes': teacher.successes,
        'history_best_fit_param': f.history_best_fit_param,
        'history_best_fit_value': f.history_best_fit_value,
    }


def main(student_model=None, teacher_model=None,
         student_param=None,
         n_item=60, grades=(1, ), t_max=2000,
         time_out=5,
         n_cpu=mp.cpu_count()-1,
         normalize_similarity=True, force=False, plot_fig=True,
         init_eval=3, verbose=True
         ):

    if student_model is None:
        student_model = ActRMeaning

    if student_param is None:
        student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02}

    if teacher_model is None:
        teacher_model = AvyaTeacher

    extension = \
        f'full_stack_' \
        f'{teacher_model.__name__}_{student_model.__name__}_' \
        f'{dic2string(student_param)}_' \
        f'ni_{n_item}_grade_{grades}_tmax_{t_max}_' \
        f'norm_{normalize_similarity}_' \
        f'time_out_{time_out}'

    bkp_file = os.path.join('bkp', 'full_stack', f'{extension}.p')

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
            n_cpu=n_cpu,
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
    parser.add_argument('--no_fig', '-n', action='store_true', default=True,
                        dest='no_fig',
                        help='Do not create fig')

    parser.add_argument('--n_cpu', '-c', default=mp.cpu_count(),
                        dest='n_cpu',
                        help='Number of cpu to use', type=int)

    args = parser.parse_args()
    main(plot_fig=not args.no_fig, n_cpu=args.n_cpu)
