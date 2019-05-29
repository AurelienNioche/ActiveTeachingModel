import os
# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

# Your application specific imports
from task.models import Kanji

import pickle

import numpy as np

from task.parameters import N_POSSIBLE_REPLIES

from simulation.task import Task


BKP_FOLDER = "bkp_learning_material"

os.makedirs(BKP_FOLDER, exist_ok=True)


class TrackingTeacher:

    def __init__(self, n_item=20, t_max=100, grade=1, handle_similarities=True, verbose=False):

        assert n_item >= N_POSSIBLE_REPLIES, \
            "The number of items have to be superior to the number of possible replies"

        self.tk = Task(n_kanji=n_item, t_max=t_max, grade=grade, compute_connections=handle_similarities,
                       generate_full_task=False)

        self.verbose = verbose

        self.questions = []
        self.replies = []
        self.successes = []

        self.agent = None

    def ask(self):

        raise NotImplementedError("Tracking Teacher is a meta-class. Ask method need to be overridden")

    def get_possible_replies(self, question):

        # Select randomly possible replies, including the correct one
        all_replies = list(range(self.tk.n_item))
        all_replies.remove(question)
        possible_replies = [question, ] + list(np.random.choice(all_replies, size=N_POSSIBLE_REPLIES-1, replace=False))
        possible_replies = np.array(possible_replies)
        np.random.shuffle(possible_replies)
        return possible_replies

    def teach(self, agent):

        self.agent = agent

        for _ in range(self.tk.t_max):
            question, possible_replies = self.ask()

            reply = agent.decide(question=question, possible_replies=possible_replies)
            agent.learn(question=question)

            # For backup
            self.questions.append(question)
            self.replies.append(reply)
            self.successes.append(reply == question)  # We assume that the matching is (0,0), (1, 1), (n, n)

        return self.questions, self.replies, self.successes
