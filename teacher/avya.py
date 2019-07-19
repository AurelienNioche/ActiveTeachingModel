import copy

import numpy as np

# import random
from teacher.metaclass import GenericTeacher


class AvyaTeacher(GenericTeacher):

    def __init__(self, n_item=20, t_max=200, grade=1, handle_similarities=True,
                 iteration=0, learn_threshold=0.95, forgot_threshold=0.85,
                 represent_learnt=2, represent_learning=1, represent_unseen=0,
                 verbose=False):
        """
        :param iteration: current iteration number.
            0 at first iteration.
        :param learn_threshold: p_recall(probability of recall) threshold after
            which an item is learnt
        :param forgot_threshold: As learn_threshold but on the verge of being
            learnt
        """

        super().__init__(n_item=n_item, t_max=t_max, grade=grade,
                         handle_similarities=handle_similarities,
                         verbose=verbose)

        self.iteration = iteration
        self.learn_threshold = learn_threshold
        self.forgot_threshold = forgot_threshold
        self.represent_learnt = represent_learnt
        self.represent_learning = represent_learning
        self.represent_unseen = represent_unseen

        self.learning_progress = np.zeros(n_item)
        # :param learning_progress: list of size n_items containing:
        #     * param represent_learnt: at i^th index when i^th item is learnt
        #     * param represent_learning: at i^th index when i^th item is seen
        #     at least once but hasn't been learnt
        #     * param represent_unseen: at i^th index when i^th item is unseen

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

    def update_sets(self, agent, n_items):
        """
        :param agent:
        :param n_items:

        Updates the learning progress list after every iteration.
        Follows the following rules:
            1. When the probability of recall of any item is greater than
            learn_threshold then update said item to represent_learnt value in
            learning_progress array.
            2. If the item is unseen until (current iteration - 1) then update
            item value to represent_learning value in learning_progress array.
        """
        for i in range(n_items):
            if agent.p_recall(i) > self.learn_threshold:
                self.learning_progress[i] = self.represent_learnt

        if self.iteration > 0:
            shown_item = self.questions[self.iteration - 1]
            if self.learning_progress[shown_item] == self.represent_unseen:
                self.learning_progress[shown_item] = self.represent_learning

    def get_parameters(self, n_items, agent):
        """
        Function to calculate Usefulness of items
        :var recall_arr: list of integers size n_items (ith index has current
        probability of recall of ith item)
        :var recall_next_arr: matrix of integers size n_items*n_items ((i,j)th
        index is probability of recalling i with j learnt by learner of recall
        of ith item)
        :return the following variables:
            * :var relative: matrix of integers size n_items*n_items((i,j)th index
                has relative amount by which knowing j helps i
            * :var usefulness: list of integers that stores the usefulness of
                teaching the ith item now.
        """

        recall_arr = np.zeros(n_items)
        recall_next_arr = np.zeros((n_items, n_items))
        relative = np.zeros((n_items, n_items))
        usefulness = np.zeros(n_items)

        for i in range(n_items):
            recall_arr[i] = agent.p_recall(i)
            for j in range(n_items):
                if j != i:
                    agent.learn(j)
                    recall_next_arr[i][j] = agent.p_recall(i)
                    relative[i][j] = recall_next_arr[i][j] - recall_arr[i]
                    agent.unlearn()
                usefulness[j] += relative[i][j]
        return relative, usefulness

    def get_useful(self, questions, agent, n_items):
        """Rule 3: find the most useful item"""
        relative, usefulness = self.get_parameters(n_items, agent)
        max_ind = None
        max_val = float('-inf')
        for i in range(0, n_items):
            if self.learning_progress[i] != self.represent_learnt:
                if usefulness[i] > max_val:
                    if questions[self.iteration - 1] != i:
                        max_val = usefulness[i]
                        max_ind = i
        if max_ind is None:
            print("All items learnt by Learner")
            max_ind = 0

        new_question = max_ind
        self.update_sets(agent, n_items)
        self.iteration += 1
        return new_question

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
        """
            Function implements 3 Rules in order:
            1. Teach a learnt item whose p_recall slipped below learn_threshold 
            2. Teach a learning item with p_recall above forgot threshold
            3. Teach the most useful item 
        """

        if self.iteration > 0:
            for i in range(n_items):
                # Rule1:
                if self.learning_progress[i] == self.represent_learnt:
                    if agent.p_recall(i) < self.learn_threshold:
                        if questions[self.iteration - 1] != i:
                            new_question = i
                            self.update_sets(agent, n_items)
                            self.iteration += 1
                            return new_question
                # Rule2:
                elif self.learning_progress[i] == self.represent_learning:
                    if agent.p_recall(i) > self.forgot_threshold:
                        if questions[self.iteration - 1] != i:
                            new_question = i
                            self.update_sets(agent, n_items)
                            self.iteration += 1
                            return new_question

        new_question = self.get_useful(questions, agent, n_items)
        return new_question


