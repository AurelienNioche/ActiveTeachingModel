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

from learner.rl import QLearner
from learner.act_r import ActR
from learner.act_r_custom import ActRMeaning, ActRGraphic, ActRPlus, ActRPlusPlus

from task.parameters import N_POSSIBLE_REPLIES

import similarity_graphic.measure
import similarity_semantic.measure

from fit import fit

from tqdm import tqdm


def create_questions(t_max=100, n_kanji=20, grade=1, verbose=False):

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
            list(np.random.choice(meaning_without_correct, size=N_POSSIBLE_REPLIES - 1, replace=False))

        # Randomize the order of possible replies
        np.random.shuffle(possible_replies)

        question_list.append(question)
        correct_answer_list.append(correct_answer)
        possible_replies_list.append(possible_replies)

    if verbose:
        print(f"Kanjis used are: {kanji}\n")
        print(f"Meanings used are: {meaning}\n")

    return kanji, meaning, question_list, correct_answer_list, possible_replies_list


def get_simulated_data(model, parameters, kanjis, meanings, questions_list, possible_replies_list,
                       c_graphic=None, c_semantic=None, verbose=False):

    n_items = len(kanjis)
    t_max = len(possible_replies_list)

    task_features = {
        'n_items': n_items,
        't_max': t_max,
        'n_possible_replies': N_POSSIBLE_REPLIES,
        'c_graphic': c_graphic,
        'c_semantic': c_semantic
    }

    agent = model(parameters=parameters, task_features=task_features, verbose=verbose)

    questions = np.asarray([kanjis.index(q) for q in questions_list], dtype=int)
    replies = np.zeros(t_max, dtype=int)
    success = np.zeros(t_max, dtype=int)
    possible_replies = np.zeros((t_max, N_POSSIBLE_REPLIES), dtype=int)

    if verbose:
        print(f"Generating data with model {model.__name__}...")
        gen = tqdm(range(t_max))
    else:
        gen = range(t_max)

    for t in gen:
        q = questions[t]
        for i in range(N_POSSIBLE_REPLIES):
            possible_replies[t, i] = meanings.index(possible_replies_list[t][i])

        r = agent.decide(question=q, possible_replies=possible_replies[t])
        agent.learn(question=q)

        replies[t] = r
        success[t] = q == r

    return questions, replies, n_items, possible_replies, success


def create_simulated_data(model=None, parameters=None, t_max=300, n_kanji=30, grade=1, verbose=False, force=False,
                          plot_success=False):

    if model is None and parameters is None:
        model = model
        parameters = {"alpha": 0.1, "tau": 0.01}

    print(f"Simulating {model.__name__} with parameters: "
          f"{''.join(f'{k}={round(parameters[k], 3)}, ' for k in sorted(parameters.keys()))[:-2]}...", end=' ')

    kanjis, meanings, question_list, correct_answer_list, possible_replies_list = \
        create_questions(t_max=t_max, n_kanji=n_kanji, grade=grade, verbose=verbose)

    c_graphic = similarity_graphic.measure.get(kanjis, verbose=verbose)
    c_semantic = similarity_semantic.measure.get(meanings, verbose=verbose)

    questions, replies, n_items, possible_replies, success = get_simulated_data(
        model=model,
        parameters=parameters,
        kanjis=kanjis, meanings=meanings,
        questions_list=question_list,
        possible_replies_list=possible_replies_list,
        c_graphic=c_graphic, c_semantic=c_semantic,
        verbose=verbose
    )

    if plot_success:
        plot.success.curve(success, fig_name=f'simulated_{model.__name__}.pdf')

    print("Done!\n")

    return questions, replies, n_items, possible_replies, c_graphic, c_semantic


def create_and_fit_simulated_data(model=None, parameters=None, t_max=300, n_kanji=30, grade=1,
                                  use_p_correct=False, verbose=False, force=False, fit_method='tpe'):

    questions, replies, n_items, possible_replies, c_graphic, c_semantic = create_simulated_data(
        model=model, parameters=parameters, t_max=t_max, n_kanji=n_kanji,
        grade=grade, verbose=verbose, force=force)

    f = fit.Fit(questions=questions, replies=replies, possible_replies=possible_replies, n_items=n_items,
                c_graphic=c_graphic, c_semantic=c_semantic, use_p_correct=use_p_correct, method=fit_method)
    # print()
    # f.rl()
    # print()
    # f.act_r()
    f.act_r_meaning()
    f.act_r_graphic()
    # best_param, mean_p, lls, bic = f.act_r_meaning()
    # f.act_r_plus()
    # f.act_r_plus_plus()
    print("\n" + ''.join("_" * 10) + "\n")


def demo():

    for m, p in (
            (QLearner, {"alpha": 0.05, "tau": 0.01}),
            (ActR, {"d": 0.5, "tau": 0.05, "s": 0.4}),
            (ActRPlus, {"d": 0.5, "tau": 0.05, "s": 0.4, "m": 0.3, "g": 0.7}),
            (ActRPlusPlus, {"d": 0.5, "tau": 0.05, "s": 0.4, "m": 0.3, "g": 0.7,
                            "g_mu": 0.5, "g_sigma": 1.2, "m_mu": 0.2, "m_sigma": 2}),
    ):
        create_and_fit_simulated_data(model=m, parameters=p, use_p_correct=False, verbose=True)


def main():

    m, p = ActRMeaning, {"d": 0.5, "tau": 0.05, "s": 0.05, "m": 0.7}
    # create_and_fit_simulated_data(learner=m, parameters=p, verbose=False, use_p_correct=True, fit_method='de',
    #                               t_max=600)

    create_simulated_data(model=m, parameters=p, verbose=False, t_max=300, n_kanji=30, plot_success=True)
    # m, p = ActRGraphic, {"d": 0.5, "tau": 0.5, "s": 0.5, "g": 0.7}
    # create_and_fit_simulated_data(learner=m, parameters=p, verbose=False, use_p_correct=True, fit_method='de',
    #                               t_max=600)
    # create_and_fit_simulated_data(learner=learner, parameters=parameters, verbose=True, t_max=500, n_kanji=70)


if __name__ == "__main__":

    main()