import numpy as np

from task.parameters import N_POSSIBLE_REPLIES
from simulation.task import Task


class GenericTeacher:

    def __init__(self, n_item=20, t_max=100, grade=1, seed=123,
                 handle_similarities=True, normalize_similarity=False,
                 verbose=False):

        assert n_item >= N_POSSIBLE_REPLIES, \
            f"The number of items have to be superior to the number of possible replies " \
            f"(set to {N_POSSIBLE_REPLIES} in 'task/parameters.py')"

        assert grade != 1 or n_item <= 79, \
            "The number of items has to be inferior to 80 if" \
            "selected grade is 1."

        self.tk = Task(n_kanji=n_item, t_max=t_max, grade=grade, seed=seed,
                       compute_similarity=handle_similarities,
                       normalize_similarity=normalize_similarity,
                       generate_full_task=False, verbose=verbose)

        self.verbose = verbose

        self.questions = []
        self.replies = []
        self.successes = []

        self.agent = None

    def ask(self):

        raise NotImplementedError("Tracking Teacher is a meta-class."
                                  "Ask method need to be overridden")

    def get_possible_replies(self, question):

        # Select randomly possible replies, including the correct one
        all_replies = list(range(self.tk.n_item))
        all_replies.remove(question)
        possible_replies =\
            [question, ] + list(np.random.choice(
                all_replies, size=N_POSSIBLE_REPLIES-1, replace=False))
        possible_replies = np.array(possible_replies)
        np.random.shuffle(possible_replies)
        return possible_replies

    def teach(self, agent):

        self.agent = agent

        for _ in range(self.tk.t_max):
            question, possible_replies = self.ask()

            reply = agent.decide(question=question,
                                 possible_replies=possible_replies)
            agent.learn(question=question)

            # For backup
            self.questions.append(question)
            self.replies.append(reply)
            self.successes.append(reply == question)
            # We assume that the matching is (0,0), (1, 1), (n, n)

        return self.questions, self.replies, self.successes
