import copy

import numpy as np

from teacher.metaclass import GenericTeacher


class AvyaTeacher(GenericTeacher):
    """
    :param iteration: current iteration number.
        0 at first iteration.
    :param learnt_threshold: p_recall(probability of recall) threshold after
        which an item is learnt.
    :param forgot_threshold: As learn_threshold but on the verge of being
        learnt.
    """

    def __init__(self, n_item=20, t_max=200, grades=(1, ),
                 handle_similarities=True, normalize_similarity=False,
                 iteration=0, learnt_threshold=0.95, forgot_threshold=0.85,
                 represent_learnt=2, represent_learning=1, represent_unseen=0,
                 verbose=False):

        super().__init__(n_item=n_item, t_max=t_max, grades=grades,
                         handle_similarities=handle_similarities,
                         normalize_similarity=normalize_similarity,
                         verbose=verbose)

        self.iteration = iteration
        self.learn_threshold = learnt_threshold
        self.forgot_threshold = forgot_threshold
        self.represent_learnt = represent_learnt
        self.represent_learning = represent_learning
        self.represent_unseen = represent_unseen

        self.learning_progress = np.zeros(n_item)
        # :param learning_progress: will contain:
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

    def update_sets(self, recall_arr, n_items):
        """
        :param recall_arr: list of integers size n_items (i^th index has the
            probability of recall of i^th item)
        :param n_items: n = Number of items included (0 ... n-1)

        Updates the learning progress list after every iteration.
        Implements the following update:
            1. When the probability of recall of any item is greater than
            learn_threshold then update said item to represent_learnt value in
            learning_progress array.
            2. If the previously shown item was unseen until the last iteration
             then update item value to represent_learning value in
             learning_progress array.
        """
        for i in range(n_items):
            if recall_arr[i] > self.learn_threshold:
                self.learning_progress[i] = self.represent_learnt

        if self.iteration > 0:
            shown_item = self.questions[self.iteration - 1]
            if self.learning_progress[shown_item] == self.represent_unseen:
                self.learning_progress[shown_item] = self.represent_learning

    @staticmethod
    def get_parameters(n_items, agent):
        """
        :param agent: agent object (RL, ACT-R, ...) that implements at least
            the following methods:
            * p_recall(item): takes index of a question and gives the
                probability of recall for the agent in current state.
            * learn(item): strengthen the association between a kanji and
                its meaning.
            * unlearn(): cancel the effect of the last call of the learn
                method.
        :param n_items: as before.
        :var recall_arr: list of integers size n_items (ith index has current
        probability of recall of ith item).
        :var recall_next_arr: matrix of integers size n_items*n_items ((i,j)th
        index is probability of recalling i with j learnt).
        :var relative: matrix of integers size n_items*n_items((i,j)th index
                has relative value by which knowing j helps i.
        :return the following variables:
            * :var usefulness: list of integers that stores the usefulness of
                teaching the ith item in current iteration.
            * :var recall_arr: list of integers size n_items ( i^th index has
            the probability of recall of i^th item).

        Calculate Usefulness of items
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
        return usefulness, recall_arr

    def get_slipping(self, taboo, recall_arr, n_items):
        """
        :param taboo: Integer value from range(0 to n_items) is the item shown
        in last iteration.
        :param recall_arr: as before.
        :param n_items: as before.

        Rule 1: Find a learnt item whose p_recall slipped below learn_threshold
        """
        for i in range(n_items):
            if self.learning_progress[i] == self.represent_learnt:
                if recall_arr[i] < self.learn_threshold:
                    if taboo != i:
                        new_question = i
                        self.update_sets(recall_arr, n_items)
                        self.iteration += 1
                        return new_question
        return None

    def get_almost_learnt(self, taboo, recall_arr, n_items):
        """Rule 2: Find learning item with p_recall above forgot threshold"""
        for i in range(n_items):
            if self.learning_progress[i] == self.represent_learning:
                if recall_arr[i] > self.forgot_threshold:
                    if taboo != i:
                        new_question = i
                        self.update_sets(recall_arr, n_items)
                        self.iteration += 1
                        return new_question
        return None

    def get_useful(self, taboo, recall_arr, usefulness, n_items):
        """
        :param taboo: as before.
        :param recall_arr: as before.
        :param usefulness: list of integers that stores the usefulness of
                teaching the ith item in current iteration.
        :param n_items: as before.

        Rule 3: Find the most useful item.
        """
        max_ind = None
        max_val = float('-inf')
        for i in range(0, n_items):
            if self.learning_progress[i] != self.represent_learnt:
                if usefulness[i] > max_val:
                    if taboo != i:
                        max_val = usefulness[i]
                        max_ind = i
        if max_ind is None:
            # print("All items learnt by Learner")
            # find the least learnt item
            result = np.where(recall_arr == np.amin(recall_arr))
            max_ind = result[0][0]
        new_question = max_ind
        self.update_sets(recall_arr, n_items)
        self.iteration += 1
        return new_question

    def get_next_node(self, questions, agent, n_items):
        """
        :param questions: list of integers (index of questions).
            Empty at first iteration.
        :param agent: as before.
        :param n_items:
            * Number of items included (0 ... n-1).
        :return: integer (index of the question to ask).

        Function implements 3 Rules in order.
        """
        new_question = None
        taboo = None
        usefulness, recall_arr = self.get_parameters(n_items, agent)
        if self.iteration > 0:
            taboo = questions[self.iteration-1]
            new_question = self.get_slipping(taboo, recall_arr, n_items)
            if new_question is None:
                new_question = self.get_almost_learnt(taboo, recall_arr,
                                                      n_items)

        if new_question is None:
            new_question = self.get_useful(taboo, recall_arr, usefulness,
                                           n_items)
        return new_question
