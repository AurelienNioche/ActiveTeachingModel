import copy

import numpy as np

from teacher.metaclass import GenericTeacher


class AvyaTeacher(GenericTeacher):
    """
    :param learnt_threshold: p_recall(probability of recall) threshold after
        which an item is learnt.
    :param forgot_threshold: As learn_threshold but on the verge of being
        learnt.
    """

    def __init__(self, n_item=20, t_max=200, grades=(1,),
                 handle_similarities=True, normalize_similarity=False,
                 learnt_threshold=0.95, forgot_threshold=0.85,
                 represent_learnt=2, represent_learning=1, represent_unseen=0,
                 verbose=False):

        super().__init__(n_item=n_item, t_max=t_max, grades=grades,
                         handle_similarities=handle_similarities,
                         normalize_similarity=normalize_similarity,
                         verbose=verbose)

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

    def update_sets(self, recall_arr):
        """
        :param recall_arr: list of integers size n_item
        (i^th index has the probability of recall of i^th item)

        Updates the learning progress list after every iteration.
        Implements the following update:
            1. When the probability of recall of any item is greater than
            learn_threshold then update said item to represent_learnt value in
            learning_progress array.
            2. If the previously shown item was unseen until the last iteration
             then update item value to represent_learning value in
             learning_progress array.
        """
        for i in range(self.tk.n_item):
            if recall_arr[i] > self.learn_threshold:
                self.learning_progress[i] = self.represent_learnt

        if self.t > 0:
            shown_item = self.questions[self.t - 1]
            if self.learning_progress[shown_item] == self.represent_unseen:
                self.learning_progress[shown_item] = self.represent_learning

    def get_parameters(self, agent):
        """
        :param agent: agent object (RL, ACT-R, ...) that implements at least
            the following methods:
            * p_recall(item): takes index of a question and gives the
                probability of recall for the agent in current state.
            * learn(item): strengthen the association between a kanji and
                its meaning.
            * unlearn(): cancel the effect of the last call of the learn
                method.
        :var recall_arr: list of integers size n_item (ith index has current
        probability of recall of ith item).
        :var recall_next_arr: matrix of integers size n_item*n_item ((i,j)th
        index is probability of recalling i with j learnt).
        :var relative: matrix of integers size n_item*n_item((i,j)th index
                has relative value by which knowing j helps i.
        :return the following variables:
            * :var usefulness: list of integers that stores the usefulness of
                teaching the ith item in current iteration.
            * :var recall_arr: list of integers size n_item ( i^th index has
            the probability of recall of i^th item).

        Calculate Usefulness of items
        """

        recall_arr = np.zeros(self.tk.n_item)
        recall_next_arr = np.zeros((self.tk.n_item, self.tk.n_item))
        relative = np.zeros((self.tk.n_item, self.tk.n_item))
        usefulness = np.zeros(self.tk.n_item)

        for i in range(self.tk.n_item):
            recall_arr[i] = agent.p_recall(i)
            for j in range(self.tk.n_item):
                if j != i:
                    agent.learn(j)
                    recall_next_arr[i][j] = agent.p_recall(i)
                    relative[i][j] = recall_next_arr[i][j] - recall_arr[i]
                    agent.unlearn()
                usefulness[j] += relative[i][j]
        return usefulness, recall_arr

    def get_slipping(self, taboo, recall_arr):
        """
        :param taboo: Integer value from range(0 to n_item) is the item shown
        in last iteration.
        :param recall_arr: as before.

        Rule 1: Find a learnt item whose p_recall slipped below learn_threshold
        """
        for i in range(self.tk.n_item):
            if self.learning_progress[i] == self.represent_learnt:
                if recall_arr[i] < self.learn_threshold:
                    if taboo != i:
                        new_question = i
                        return new_question
        return None

    def get_almost_learnt(self, taboo, recall_arr):
        """Rule 2: Find learning item with p_recall above forgot threshold"""
        for i in range(self.tk.n_item):
            if self.learning_progress[i] == self.represent_learning:
                if recall_arr[i] > self.forgot_threshold:
                    if taboo != i:
                        new_question = i
                        return new_question
        return None

    def get_useful(self, taboo, recall_arr, usefulness):
        """
        :param taboo: as before.
        :param recall_arr: as before.
        :param usefulness: list of integers that stores the usefulness of
                teaching the ith item in current iteration.

        Rule 3: Find the most useful item.
        """
        max_ind = None
        max_val = float('-inf')
        for i in range(self.tk.n_item):
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
        return new_question

    def _get_next_node(self, agent=None):
        """
        :param agent: as before.
        :return: integer (index of the question to ask).

        Function implements 3 Rules in order.
        """

        agent = copy.deepcopy(agent)

        new_question = None
        taboo = None
        usefulness, recall_arr = self.get_parameters(agent)

        if self.t > 0:
            self.update_sets(recall_arr)
            taboo = self.questions[self.t - 1]
            new_question = self.get_slipping(taboo, recall_arr)
            if new_question is None:
                new_question = self.get_almost_learnt(taboo, recall_arr)

        if new_question is None:
            new_question = self.get_useful(taboo, recall_arr, usefulness)

        return new_question
