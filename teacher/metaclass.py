import numpy as np

from tqdm import tqdm

from task.parameters import N_POSSIBLE_REPLIES
from simulation.task import Task


class GenericTeacher:

    def __init__(self, n_item=20, t_max=100, grades=(1, ), seed=123,
                 handle_similarities=True, normalize_similarity=False,
                 verbose=False):

        assert n_item >= N_POSSIBLE_REPLIES, \
            f"The number of items have to be " \
            f"superior to the number of possible replies " \
            f"(set to {N_POSSIBLE_REPLIES} in 'task/parameters.py')"

        assert grades != (1, ) or n_item <= 79, \
            "The number of items has to be inferior to 80 if" \
            "selected grade is 1."

        self.tk = Task(n_kanji=n_item, t_max=t_max, grades=grades, seed=seed,
                       compute_similarity=handle_similarities,
                       normalize_similarity=normalize_similarity,
                       generate_full_task=False, verbose=verbose)

        self.verbose = verbose

        self.questions = np.ones(t_max, dtype=int) * -1
        self.replies = np.ones(t_max, dtype=int) * -1
        self.successes = np.ones(t_max, dtype=bool)

        self.seen = np.zeros((n_item, t_max), dtype=bool)

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

    def teach(self, agent, verbose=None):

        self.agent = agent

        iterator = tqdm(range(self.tk.t_max)) \
            if verbose else range(self.tk.t_max)

        for t in iterator:
            question, possible_replies = self.ask()

            reply = agent.decide(question=question,
                                 possible_replies=possible_replies)
            agent.learn(question=question)

            # Update the count of item seen
            if t > 0:
                self.seen[:, t] = self.seen[:, t - 1]
            self.seen[question, t] = True

            # For backup
            self.questions[t] = question
            self.replies[t] = reply
            self.successes[t] = reply == question
            # We assume that the matching is (0,0), (1, 1), (n, n)

        return self.questions, self.replies, self.successes
