import copy

# import random
from teacher.metaclass import GenericTeacher
import numpy as np


class AvyaTeacher(GenericTeacher):

    def __init__(self, n_item=20, t_max=200, grade=1, handle_similarities=True,
                 verbose=False):

        super().__init__(n_item=n_item, t_max=t_max, grade=grade,
                         handle_similarities=handle_similarities,
                         verbose=verbose)
        self.learned = np.zeros(n_item)
        self.learn_threshold = 0.95
        self.forgot_threshold = 0.85
        self.count = 0

    def ask(self):

        question = self.get_next_node(
            questions=self.questions,
            agent=copy.deepcopy(self.agent),
            n_items=self.tk.n_item
        )
        possible_replies = self.get_possible_replies(question)

        if self.verbose:
            print(f"Question chosen: {self.tk.kanji[question]}; "
                  f"correct answer: {self.tk.meaning[question]}; "
                  f"possible replies: {self.tk.meaning[possible_replies]};")

        return question, possible_replies

    # Learnt character set is represented by 2,Being learnt character set by 1,
    # Unseen character set as 0
    def update_sets(self, agent, n_items):
        for k in range(n_items):
            if agent.p_recall(k) > self.learn_threshold:
                self.learned[k] = 2

        if self.count > 0:
            if self.learned[self.questions[self.count-1]] == 0:
                self.learned[self.questions[self.count - 1]] = 1

    # calculate usefulness and relative parameters.
    def parameters(self, n_items, agent):
        recall = [0] * n_items
        # recall[i] represents probability of recalling kanji[i] at current
        # instant
        usefulness = [0] * n_items
        recall_next = [[0] * n_items] * n_items  # recall_next[i][j] is prob of
        # recalling i with j learnt.
        relative = [[0] * n_items] * n_items  # relative amount by which
        # knowing j helps i
        for item in range(n_items):
            recall[item] = agent.p_recall(item)
            for item2 in range(n_items):
                if item2 != item:
                    agent.learn(item2)
                    recall_next[item][item2] = agent.p_recall(item)
                    relative[item][item2] = recall_next[item][item2]\
                                            - recall[item]
                    agent.unlearn()
                usefulness[item2] += relative[item][item2]
        return relative, usefulness

    def get_next_node(self, questions, agent, n_items):
        """
        :param questions: list of integers (index of questions).
            Empty at first iteration.
        :param agent: agent object (RL, ACT-R, ...) that implements at
            least the following methods:
            * p_recall(item): takes index of a question and gives the
                probability of recall for the agent in current state.
            * learn(item): strengthen the association between a kanji and
                its meaning.
            * unlearn(): cancel the effect of the last call of the learn
                method.
        :param n_items:
            * Number of items included (0 ... n-1)
        :return: integer (index of the question to ask)
        """

        relative, usefulness = self.parameters(n_items, agent)
        if self.count > 0:
            # Rule1: don't let a learnt kanji slip out of threshold
            for i in range(n_items):
                if self.learned[i] == 2:
                    if agent.p_recall(i) < self.learn_threshold:
                        if questions[self.count-1] != i:
                            new_question = i
                            self.update_sets(agent, n_items)
                            self.count += 1
                            return new_question
            # Rule2: Bring an almost learnt Kanji to learnt set.
            for i in range(n_items):
                if self.learned[i] == 1:
                    if agent.p_recall(i) > self.forgot_threshold:
                        if questions[self.count-1] != i:
                            new_question = i
                            self.update_sets(agent, n_items)
                            self.count += 1
                            return new_question

        # Rule3: find the most useful kanji.
        max_ind = -1
        max_val = -10000
        for i in range(0, n_items):
            if self.learned[i] < 2:
                if usefulness[i] > max_val:
                    if questions[self.count - 1] != i:
                        max_val = usefulness[i]
                        max_ind = i

        new_question = max_ind
        self.update_sets(agent, n_items)

        # Update the iteration index
        self.count += 1
        return new_question
