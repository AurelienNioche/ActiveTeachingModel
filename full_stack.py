import numpy as np
from tqdm import tqdm

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
from fit.bayesian import BayesianFit

import warnings


def run(student_model, teacher_model, student_param,
        n_item, grades, t_max, normalize_similarity):

    teacher = teacher_model(t_max=t_max, n_item=n_item,
                            normalize_similarity=normalize_similarity,
                            grades=grades)

    learner = student_model(param=student_param, tk=teacher.tk)

    iterator = tqdm(range(t_max))  # if verbose else range(t_max)

    model_learner = student_model(
        tk=teacher.tk,
        param=student_model.generate_random_parameters()
    )

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

        # print('Q&R',question, reply)

        # Update the model of the learner

        data_view = Data(n_items=n_item,
                         questions=teacher.questions[:t+1],
                         replies=teacher.replies[:t+1],
                         possible_replies=teacher.possible_replies[:t+1, :])

        f = BayesianFit(model=student_model, tk=teacher.tk, data=data_view)
        fit_r = f.evaluate(verbose=False)
        if fit_r is not None:
            print(t, 'FIT')
            model_learner.set_parameters(fit_r['best_param'])
        else:
            print(t, 'TA MERE')

        # model_learner.set_parameters(
        #     {tup[0]: getattr(learner, tup[0]) for tup in learner.bounds}
        # )

        model_learner.learn(question=question)
        # We assume that the matching is (0,0), (1, 1), (n, n)
        # print(model_learner.d)
        # print("IIII")

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


def main():

    r = run(
        student_model=ActRMeaning, teacher_model=AvyaTeacher,
        student_param={"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02},
        n_item=30, grades=(1, ), t_max=300, normalize_similarity=True)

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

    save_fig(f"fullstack.pdf")


if __name__ == '__main__':
    main()
