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
        self.successes = np.zeros(t_max, dtype=bool)
        self.possible_replies = np.zeros((t_max, self.tk.n_possible_replies),
                                         dtype=int)

        self.seen = np.zeros((n_item, t_max), dtype=bool)

        self.t = 0

    def ask(self, agent=None, make_learn=True):

        # print("___ Question ___")
        # print(self.questions)
        # print(agent.questions)

        question = self._get_next_node(agent=agent)
        possible_replies = self._get_possible_replies(question)

        if make_learn:
            reply = agent.decide(question=question,
                                 possible_replies=possible_replies)
            agent.learn(question=question)

            self.register_question_and_reply(
                reply=reply,
                question=question,
                possible_replies=possible_replies)

        return question, possible_replies

    def register_question_and_reply(self, reply, question, possible_replies):

        # Update the count of item seen
        if self.t > 0:
            self.seen[:, self.t] = self.seen[:, self.t - 1]
        self.seen[question, self.t] = True

        # For backup
        self.questions[self.t] = question
        self.replies[self.t] = reply
        self.successes[self.t] = reply == question
        self.possible_replies[self.t] = possible_replies

        if self.verbose:
            print(f"Question chosen: {self.tk.kanji[question]}; "
                  f"correct answer: {self.tk.meaning[question]}; "
                  f"possible replies: {self.tk.meaning[possible_replies]};")

        self.t += 1

    def _get_next_node(self, agent=None):
        raise NotImplementedError(f"{type(self).__name__} is a meta-class."
                                  "This method need to be overridden")

    def _get_possible_replies(self, question):

        # Select randomly possible replies, including the correct one
        all_replies = list(range(self.tk.n_item))
        all_replies.remove(question)

        possible_replies =\
            [question, ] + list(np.random.choice(
                all_replies, size=N_POSSIBLE_REPLIES-1, replace=False))
        possible_replies = np.array(possible_replies)
        np.random.shuffle(possible_replies)
        return possible_replies

    def teach(self, agent=None):

        tqdm.write("Teaching...")

        iterator = tqdm(range(self.tk.t_max)) \
            if not self.verbose else range(self.tk.t_max)

        for _ in iterator:
            self.ask(agent=agent)

        return self.questions, self.replies, self.successes
