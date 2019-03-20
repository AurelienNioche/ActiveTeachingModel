import os
# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

# Your application specific imports
from task.models import Kanji

import numpy as np

import plot.success

from model.learner import QLearner, ActRLearner, ActRPlusLearner, ActRPlusPlusLearner

from task.parameters import n_possible_replies

import graphic_similarity.measure
import semantic_similarity.measure

from fit import fit


def create_questions(t_max=100, n_kanji=20, grade=1):

    # Seed
    np.random.seed(123)

    # Select n kanji among first grade
    k = list(Kanji.objects.filter(grade=grade).order_by('id'))

    while True:

        # Select randomly n kanji
        kanji_idx = np.random.choice(np.arange(len(k)), size=n_kanji, replace=False)

        # Get the meaning
        meaning = [k[kanji_idx[i]].meaning for i in range(n_kanji)]

        # Ensure that each kanji has a different meaning
        if len(np.unique(meaning)) == len(meaning):
            break

    # Get the kanji
    kanji = [k[kanji_idx[i]].kanji for i in range(n_kanji)]

    # Define probability for a kanji to be selected
    p = np.random.random(n_kanji)
    p /= p.sum()

    q_idx = np.random.choice(np.arange(n_kanji), size=t_max, p=p, replace=True)

    question_list = []
    correct_answer_list = []
    possible_replies_list = []

    for t in range(t_max):

        # Get question and correct answer
        question = kanji[q_idx[t]]
        correct_answer = meaning[q_idx[t]]

        # Select possible replies
        meaning_without_correct = meaning.copy()
        meaning_without_correct.remove(correct_answer)

        possible_replies = \
            [correct_answer, ] + \
            list(np.random.choice(meaning_without_correct, size=n_possible_replies - 1, replace=False))

        # Randomize the order of possible replies
        np.random.shuffle(possible_replies)

        question_list.append(question)
        correct_answer_list.append(correct_answer)
        possible_replies_list.append(possible_replies)

    return kanji, meaning, question_list, correct_answer_list, possible_replies_list


def get_simulated_data(model, parameters, kanjis, meanings, questions_list, possible_replies_list,
                       c_graphic=None, c_semantic=None):

    n_items = len(kanjis)
    t_max = len(possible_replies_list)

    task_features = (n_items, )

    if model in (ActRPlusLearner, ActRPlusPlusLearner):
        task_features += (t_max, n_possible_replies, c_graphic, c_semantic)

    agent = model(parameters, task_features)

    questions = np.asarray([kanjis.index(q) for q in questions_list], dtype=int)
    replies = np.zeros(t_max, dtype=int)
    success = np.zeros(t_max, dtype=int)
    possible_replies = np.zeros((t_max, n_possible_replies), dtype=int)

    for t in range(t_max):
        q = questions[t]
        r = agent.decide(question=q)
        agent.learn(question=q, reply=r)

        replies[t] = r

        for i in range(n_possible_replies):
            possible_replies[t, i] = meanings.index(possible_replies_list[t][i])
        success[t] = q == r

    return questions, replies, n_items, possible_replies, success


def create_and_fit_simulated_data(model=QLearner, parameters=(0.1, 0.01), t_max=100, n_kanji=20, grade=1):

    kanjis, meanings, question_list, correct_answer_list, possible_replies_list = \
        create_questions(t_max=t_max, n_kanji=n_kanji, grade=grade)

    c_graphic = graphic_similarity.measure.get(kanjis)
    c_semantic = semantic_similarity.measure.get(meanings)

    questions, replies, n_items, possible_replies, success = get_simulated_data(
        model=model,
        parameters=parameters,
        kanjis=kanjis, meanings=meanings,
        questions_list=question_list,
        possible_replies_list=possible_replies_list,
        c_graphic=c_graphic, c_semantic=c_semantic
    )

    plot.success.curve(success, fig_name=f'simulated_curve.pdf')

    f = fit.Fit(questions=questions, replies=replies, possible_replies=possible_replies, n_items=n_items,
                c_graphic=c_graphic, c_semantic=c_semantic)
    f.rl()
    f.act_r()
    f.act_r_plus()
    f.act_r_plus_plus()


def main():

    for m, p in (
            (QLearner, (0.5, 0.2)),
    ):
        create_and_fit_simulated_data(model=m, parameters=p)


if __name__ == "__main__":

    main()
