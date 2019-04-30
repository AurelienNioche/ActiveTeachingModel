import os
# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

# Your application specific imports
from task.models import Kanji

import similarity_graphic.measure
import similarity_semantic.measure

from solver import solver

import numpy as np

from model.rl import QLearner
from model.act_r import ActR, ActRMeaning, ActRGraphic, ActRPlus

import plot.success

from task.parameters import N_POSSIBLE_REPLIES

from tqdm import tqdm

import copy

import pickle


BKP_FOLDER = "niraj_bkp"

os.makedirs(BKP_FOLDER, exist_ok=True)


class Teacher:

    """
    Customize this as you want (except 'grade' and 'n_items' as in this case, you need a specific db
    """

    def __init__(self, n_items=20, t_max=100, grade=1):

        self.t_max = t_max
        self.n_items = n_items
        self.grade = grade

        self.kanjis, self.meanings = self.kanji_and_meaning

        self.graphic_similarity = similarity_graphic.measure.get(self.kanjis)
        self.semantic_similarity = similarity_semantic.measure.get(self.meanings)

    def ask(self, questions, successes, agent):

        """
        Especially this part
        """

        question = solver.get_next_node(
            questions=questions,
            successes=successes,
            agent=copy.deepcopy(agent),
            n_items=self.n_items
        )

        possible_replies = self.get_possible_replies(question)

        print(f"Question chosen: {self.kanjis[question]}; "
              f"correct answer: {self.meanings[question]}; "
              f"possible replies: {self.meanings[possible_replies]};")

        return question, possible_replies

    def get_possible_replies(self, question):

        # Select randomly possible replies, including the correct one
        all_replies = list(range(self.n_items))
        all_replies.remove(question)
        possible_replies = [question, ] + list(np.random.choice(all_replies, size=N_POSSIBLE_REPLIES-1))
        possible_replies = np.array(possible_replies)
        np.random.shuffle(possible_replies)
        return possible_replies

    @property
    def task_features(self):

        return {
            'n_items': self.n_items,
            't_max': self.t_max,
            'n_possible_replies': N_POSSIBLE_REPLIES,
            'c_graphic': self.graphic_similarity,
            'c_semantic': self.semantic_similarity
        }

    @property
    def kanji_and_meaning(self):

        bkp_file = f"{BKP_FOLDER}/kanji_meaning_n{self.n_items}_g{self.grade}.p"
        if os.path.exists(bkp_file):
            kanji, meaning = pickle.load(open(bkp_file, 'rb'))
            return kanji, meaning

        # Seed
        np.random.seed(123)

        # Select n kanji among first grade
        k = list(Kanji.objects.filter(grade=self.grade).order_by('id'))

        while True:

            # Select randomly n kanji
            kanji_idx = np.random.choice(np.arange(len(k)), size=self.n_items, replace=False)

            # Get the meaning
            meaning = [k[kanji_idx[i]].meaning for i in range(self.n_items)]

            # Ensure that each kanji has a different meaning
            if len(np.unique(meaning)) == len(meaning):
                break

        # Get the kanji
        kanji = [k[kanji_idx[i]].kanji for i in range(self.n_items)]

        kanji = np.asarray(kanji)
        meaning = np.asarray(meaning)

        pickle.dump((kanji, meaning), open(bkp_file, 'wb'))

        return kanji, meaning


def run_simulation(model, parameters, teacher, verbose=False):

    task_features = teacher.task_features
    agent = model(parameters=parameters, task_features=task_features, verbose=verbose)

    t_max = task_features['t_max']

    if verbose:
        print(f"Generating data with model {model.__name__}...")
        gen = tqdm(range(t_max))
    else:
        gen = range(t_max)

    questions = []
    replies = []
    successes = []

    for _ in gen:
        question, possible_replies = teacher.ask(questions=questions, successes=successes, agent=agent)

        reply = agent.decide(question=question, possible_replies=possible_replies)
        agent.learn(question=question)

        # For backup
        questions.append(question)
        replies.append(reply)
        successes.append(reply == question)  # We assume that the matching is (0,0), (1, 1), (n, n)

    return questions, replies, successes


def demo_all(t_max=100):

    """
    parameter mapping between paper (left) and code (right):
    lambda: d
    tau: tau
    theta: s
    gamma: g
    psi: m
    """

    for m, p in (
            (QLearner, {"alpha": 0.05, "tau": 0.01}),
            (ActR, {"d": 0.5, "tau": 0.05, "s": 0.4}),
            (ActRGraphic, {"d": 0.5, "tau": 0.05, "s": 0.4, "g": 0.5}),
            (ActRMeaning, {"d": 0.5, "tau": 0.05, "s": 0.5, "m": 0.7}),
            (ActRPlus, {"d": 0.5, "tau": 0.05, "s": 0.4, "m": 0.3, "g": 0.7})
    ):
        teacher = Teacher(t_max=t_max)
        run_simulation(model=m, parameters=p, teacher=teacher)


def demo(t_max=300, n_items=30):

    """
    parameter mapping between paper (left) and code (right):
    lambda: d
    tau: tau
    theta: s
    gamma: g
    psi: m
    """

    model, parameters = ActR, {"d": 0.5, "tau": 0.01, "s": 0.4}
    teacher = Teacher(t_max=t_max, n_items=n_items)
    questions, replies, successes = run_simulation(model=model, parameters=parameters, teacher=teacher)

    plot.success.curve(successes, fig_name=f'niraj_simulated_{model.__name__}_curve.pdf')
    plot.success.scatter(successes, fig_name=f'niraj_simulated_{model.__name__}_scatter.pdf')


if __name__ == "__main__":

    demo()

