from tqdm import tqdm

import os

import plot.memory_trace
import plot.success
import plot.n_seen
import plot.n_learnt

from learner.act_r_custom import ActRMeaning
from simulation.memory import p_recall_over_time_after_learning
from teacher.avya import AvyaTeacher

import matplotlib.pyplot as plt

from plot.generic import save_fig
from simulation.data import Data
# from fit.bayesian import BayesianFit
# from fit.bayesian_gpyopt import BayesianGPyOptFit
# from fit.bayesian_pygpgo import BayesianPYGPGOFit
# from fit.fit import Fit
from fit.bayesian_pygpgo import BayesianPYGPGOFit

from utils.utils import dic2string, dump, load

import argparse


def run(student_model, teacher_model, student_param,
        n_item, grades, t_max, normalize_similarity,
        max_iter):

    teacher = teacher_model(t_max=t_max, n_item=n_item,
                            normalize_similarity=normalize_similarity,
                            grades=grades)

    learner = student_model(param=student_param, tk=teacher.tk)

    iterator = tqdm(range(t_max))

    model_learner = student_model(
        tk=teacher.tk,
        param=student_model.generate_random_parameters()
    )

    f = BayesianPYGPGOFit(
        model=student_model, tk=teacher.tk,
        data=None)

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
                         questions=teacher.questions[:t+1],
                         replies=teacher.replies[:t+1],
                         possible_replies=teacher.possible_replies[:t+1, :])

        f.evaluate(max_iter=max_iter, data=data_view)

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
        'successes': teacher.successes
    }


def _plot(r, extension):

    seen = r['seen']
    p_recall = r['p_recall']
    successes = r['successes']

    # Plot...
    font_size = 10
    label_size = 8
    line_width = 1

    n_rows, n_cols = 5, 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6, 14))

    ax1 = axes[0]
    plot.memory_trace.summarize(
        p_recall=p_recall,
        ax=ax1,
        font_size=font_size,
        label_size=label_size,
        line_width=line_width,
    )

    ax2 = axes[1]
    plot.memory_trace.summarize_over_seen(
        p_recall=p_recall,
        seen=seen,
        ax=ax2,
        font_size=font_size,
        label_size=label_size,
        line_width=line_width
    )

    ax3 = axes[2]
    plot.n_learnt.curve(
        p_recall=p_recall,
        ax=ax3,
        font_size=font_size,
        label_size=label_size,
        line_width=line_width
    )

    ax4 = axes[3]
    plot.n_seen.curve(
        seen=seen,
        ax=ax4,
        font_size=font_size,
        label_size=label_size,
        line_width=line_width * 2
    )

    ax5 = axes[4]
    plot.success.curve(
        successes=successes,
        ax=ax5,
        font_size=font_size,
        label_size=label_size,
        line_width=line_width * 2
    )

    save_fig(f"fullstack_{extension}.pdf")


def main(student_model=None, teacher_model=None,
         student_param=None,
         n_item=60, grades=(1, ), t_max=2000,
         max_iter=10,
         normalize_similarity=True, force=False, plot_fig=True):

    if student_model is None:
        student_model = ActRMeaning

    if student_param is None:
        student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02}

    if teacher_model is None:
        teacher_model = AvyaTeacher

    extension = f'{teacher_model.__name__}_{student_model.__name__}_' \
        f'{dic2string(student_param)}_' \
        f'ni_{n_item}_grade_{grades}_tmax_{t_max}_' \
        f'norm_{normalize_similarity}_' \
        f'max_iter_{max_iter}'

    bkp_file = os.path.join('bkp', 'full_stack', f'{extension}.p')

    r = load(bkp_file)
    if r is None or force:
        r = run(
            student_model=student_model,
            teacher_model=teacher_model,
            student_param=student_param,
            n_item=n_item,
            grades=grades,
            t_max=t_max,
            normalize_similarity=normalize_similarity,
            max_iter=max_iter)

        dump(r, bkp_file)

    if plot_fig:
        _plot(r, extension)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no_fig', '-n', action='store_true', default=True,
                        dest='no_fig',
                        help='Do not create fig')

    args = parser.parse_args()
    main(plot_fig=not args.no_fig)
