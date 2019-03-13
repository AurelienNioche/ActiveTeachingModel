import os
import numpy as np
import scipy.optimize

# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

# Your application specific imports
from task.models import User, Question

from model.learner import QLearner, ActRLearner

from task.parameters import n_possible_replies

from utils.functions import bic

import graph.model_comparison
import graph.success


def _objective_act_r(parameters, questions, replies, n_items):

    d, tau, s = parameters

    if np.any(np.isnan(parameters)):
        return np.inf

    learner = ActRLearner(n_items=n_items, n_possible_replies=n_possible_replies, d=d, tau=tau, s=s)

    t_max = len(questions)

    p_choices = np.zeros(t_max)

    for t in range(t_max):

        question, reply = questions[t], replies[t]

        p = learner.p_choice(question=question, reply=reply)
        # if p == 0:
        #     print('p=0')
        #     return 0
        # else:
        #     print(t, p)

        p_choices[t] = p
        learner.learn(question=question, reply=reply)

    return - np.sum(np.log(p_choices))


def fit_act_r(questions, replies, n_items):

    res = scipy.optimize.minimize(
        _objective_act_r, np.array([0.5, 0.0001, 0.00001]), args=(questions, replies, n_items),
        bounds=((0.4, 0.6), (0.00001, 0.001), (0.005, 0.015)))  # method=SLSQP

    d, tau, s = res.x
    lls = res.fun

    best_param = {"d": d, "tau": tau, "s": s}
    n = len(questions)
    k = len(best_param)
    bic_v = bic(lls=lls, n=n, k=k)

    return lls, n, best_param, bic_v


def _objective_rl(parameters, questions, replies, possible_replies, n_items):

    alpha, tau = parameters

    if np.any(np.isnan(parameters)):
        return np.inf

    learner = QLearner(n_items=n_items, alpha=alpha, tau=tau)

    t_max = len(questions)

    p_choices = np.zeros(t_max)

    for t in range(t_max):

        question, reply, possible_rep = questions[t], replies[t], possible_replies[t]

        p = learner.p_choice(question=question, reply=reply, possible_replies=possible_rep)
        # if p == 0:
        #     print('p=0')
        #     return np.inf

        p_choices[t] = p
        learner.learn(question=question, reply=reply)

    return - np.sum(np.log(p_choices))


def fit_rl(questions, replies, possible_replies, n_items):

    res = scipy.optimize.minimize(
        _objective_rl, np.array([0.1, 0.01]), args=(questions, replies, possible_replies, n_items),
        bounds=((0.00, 1.00), (0.001, 0.1)))  # method=SLSQP

    alpha, tau = res.x
    lls = res.fun

    best_param ={'alpha': alpha, 'tau': tau}
    n = len(questions)
    k = len(best_param)
    bic_v = bic(lls=lls, n=n, k=k)

    return lls, n, {'alpha': alpha, 'tau': tau}, bic_v


def print_data():

    users = User.objects.all().order_by('id')

    for user in users:

        user_id = user.id

        print(f"User {user_id}")
        print("*" * 4)

        que = Question.objects.filter(user_id=user_id).order_by('t')

        for q in que:
            t = q.t
            reaction_time = int((q.time_reply - q.time_display).total_seconds() * 10 ** 3)
            success = q.reply == q.correct_answer

            to_print = f't:{t}, question: {q.question}, reply: {q.reply}, correct answer: {q.correct_answer}, ' \
                f'success: {"Yes" if success else "No"}, reaction time: {reaction_time} ms'
            print(to_print)

        print()


def get_task_features(user_id):

    question_entries = [q for q in Question.objects.filter(user_id=user_id).order_by('t')]

    items = [q.question for q in question_entries]

    kanjis = list(np.unique(items))
    meanings = []

    for k in kanjis:
        meanings.append(Question.objects.filter(user_id=user_id, question=k)[0].correct_answer)

    print(f"Kanji used: {kanjis}")
    print(f"Corresponding meanings: {meanings}")

    return question_entries, kanjis, meanings


def get_data(user_id):

    question_entries, kanjis, meanings = get_task_features(user_id=user_id)

    n_items = len(kanjis)

    t_max = question_entries[-1].t

    questions = np.zeros(t_max, dtype=int)
    replies = np.zeros(t_max, dtype=int)
    possible_replies = np.zeros((t_max, n_possible_replies), dtype=int)
    for t in range(t_max):

        questions[t] = kanjis.index(question_entries[t].question)
        replies[t] = meanings.index(question_entries[t].reply)
        for i in range(n_possible_replies):
            possible_replies[t, i] = meanings.index(getattr(question_entries[t], f'possible_reply_{i}'))

    return questions, replies, n_items, possible_replies


def main():

    users = User.objects.all().order_by('id')

    bic_data = [[], []]

    for u in users:
        user_id = u.id
        print(user_id)
        print("*" * 5)

        questions, replies, n_items, possible_replies = get_data(user_id)

        lls, n, best_param, bic_v = fit_rl(questions=questions, replies=replies, n_items=n_items, possible_replies=possible_replies)
        print(f'Alpha: {best_param["alpha"]}, Tau: {best_param["tau"]}, LLS: {lls:.2f}, BIC:{bic_v:.2f}')

        bic_data[0].append(bic_v)

        lls, n, best_param, bic_v = fit_act_r(questions=questions, replies=replies, n_items=n_items)
        print(f'd: {best_param["d"]}, Tau: {best_param["tau"]}, s:{best_param["s"]:.3f}, LLS: {lls:.2f}, BIC:{bic_v:.2f}')

        bic_data[1].append(bic_v)
        print()

    graph.model_comparison.scatter_plot(data_list=bic_data, colors=["C0", "C1"], x_tick_labels=["RL", "ACT-R - -"],
                                        f_name='model_comparison.pdf', y_label='BIC')


if __name__ == "__main__":

    main()
