# import copy

import numpy as np

from teacher.metaclass import GenericTeacher


class AvyaTeacher(GenericTeacher):

    represent_unseen = 0
    represent_learning = 1
    represent_learnt = 2

    def __init__(self, n_item=20, t_max=200, grades=(1, ),
                 handle_similarities=True, normalize_similarity=False,
                 learnt_threshold=0.95, forgot_threshold=0.85,
                 verbose=False):

        """
        :param n_item: task attribute
        :param t_max: task attribute
        :param grades: task attribute
        :param handle_similarities: task attribute
        :param normalize_similarity: task attribute
        :param learnt_threshold: p_recall(probability of recall) threshold after
        which an item is learnt.
        :param forgot_threshold: As learn_threshold but on the verge of being
        learnt.
        :param verbose: be talkative (or not)

        :var self.taboo: Integer value from range(0 to n_item)
        is the item shown in last iteration.
        :var self.p_recall: array of float of size n_item
        (ith index has current probability of recall of ith item).
        :var self.usefulness: array of floats that stores the usefulness of
                teaching the ith item in current iteration.
        :var self.recall_arr: array of integers size n_item ( i^th index has
            the probability of recall of i^th item).
        """

        super().__init__(n_item=n_item, t_max=t_max, grades=grades,
                         handle_similarities=handle_similarities,
                         normalize_similarity=normalize_similarity,
                         verbose=verbose)

        self.learn_threshold = learnt_threshold
        self.forgot_threshold = forgot_threshold

        self.learning_progress = np.zeros(n_item, dtype=int)
        # :param learning_progress: will contain:
        #     * param represent_learnt: at i^th index when i^th item is learnt
        #     * param represent_learning: at i^th index when i^th item is seen
        #     at least once but hasn't been learnt
        #     * param represent_unseen: at i^th index when i^th item is unseen

        self.p_recall = np.zeros(self.tk.n_item)
        self.usefulness = np.zeros(self.tk.n_item)

        self.items = np.arange(self.tk.n_item)

        self.question = None
        self.taboo = None
        self.not_taboo = np.ones(self.tk.n_item, dtype=bool)

        self.rule = None

    def update_sets(self):
        """

        Updates the learning progress list after every iteration.
        Implements the following update:
            1. When the probability of recall of any item is greater than
            learn_threshold then update said item to represent_learnt value in
            learning_progress array.
            2. If the previously shown item was unseen until the last iteration
             then update item value to represent_learning value in
             learning_progress array.
        """

        learnt = self.learning_progress == self.represent_learnt
        under_learnt_threshold = self.p_recall < self.learn_threshold

        # Downgrade items that are not above the threshold anymore
        self.learning_progress[learnt*under_learnt_threshold] = \
            self.represent_learning

        # Upgrade items that are above
        self.learning_progress[np.invert(under_learnt_threshold)] = \
            self.represent_learnt

        if self.t > 0 and \
                self.learning_progress[self.taboo] == self.represent_unseen:

            self.learning_progress[self.taboo] = self.represent_learning

    def _get_parameters(self, agent):
        """
        :param agent: agent object (RL, ACT-R, ...) that implements at least
            the following methods:
            * p_recall(item): takes index of a question and gives the
                probability of recall for the agent in current state.
            * learn(item): strengthen the association between a kanji and
                its meaning.
            * unlearn(): cancel the effect of the last call of the learn
                method.
        :return None

        Calculate Usefulness of items
        """

        self.usefulness[:] = 0

        for i in range(self.tk.n_item):
            p_recall_i = agent.p_recall(i)
            self.p_recall[i] = p_recall_i
            for j in range(self.tk.n_item):
                if j != i:
                    agent.learn(j)
                    p_recall_i_after_j = agent.p_recall(i)
                    agent.unlearn()

                    relative = p_recall_i_after_j - p_recall_i
                    self.usefulness[j] += relative

    def _get_selection(self, learning_state, randomize=True):

        cond = self.learning_progress == learning_state
        selection = self.items[cond * self.not_taboo]
        if randomize:
            np.random.shuffle(selection)
        return selection

    # def get_slipping(self):
    #     """
    #     Rule 1: Find a learnt item whose p_recall slipped below learn_threshold
    #     """
    #     selection = self._get_selection(learning_state=self.represent_learnt)
    #     if len(selection):
    #         p_recall = self.p_recall[selection]
    #         if p_recall.min() < self.learn_threshold:
    #             return selection[np.argmin(p_recall)]
    #     return None

    def _get_most_useful(self, learning_state):

        selection = self._get_selection(learning_state)
        if len(selection):
            return selection[np.argmax(self.usefulness[selection])]
        return None

    def rule_old_useful(self):
        """
        Rule 2: Find the most useful item among the seen items.
        """
        self.question = self._get_most_useful(self.represent_learning)

    def rule_smallest_p(self):

        poss = \
            np.where(self.p_recall == self.p_recall[self.not_taboo].min())[0]
        self.question = np.random.choice(poss)

    def rule_new_useful(self):
        """
        Rule 3: Find the most useful item among the new items.
        """
        self.question = self._get_most_useful(self.represent_unseen)

    def _find_new_question(self):

        if self.t > 0:
            # new_question = self.get_slipping()
            # if new_question is not None:
            #     if self.verbose:
            #         print('Teacher rule: Slipping rule')
            #     return new_question

            # new_question = self.get_almost_learnt()
            # if new_question not in (None, self.taboo):
            #     print('Almost learnt rule')
            #     return new_question

            self.rule_old_useful()
            if self.question is not None:
                self.rule = 'OLD useful'
                if self.verbose:
                    print('Teacher rule: Useful OLD rule')
                return

        self.rule_new_useful()
        if self.question is not None:
            self.rule = 'NEW useful'
            if self.verbose:
                print('Teacher rule: Useful NEW rule')
            return

        self.rule = 'Smallest probability'
        if self.verbose:
            print("Teacher rule: The smallest probability of recall")
        self.rule_smallest_p()

    def _get_next_node(self, agent=None):
        """
        :param agent: as before.
        :return: integer (index of the question to ask).

        Function implements 3 Rules in order.
        """

        # agent = copy.deepcopy(agent)
        self._get_parameters(agent)

        if self.t > 0:
            self.taboo = self.question
            self.question = None
            self.not_taboo[:] = self.items != self.taboo
            self.update_sets()

        self._find_new_question()

        return self.question


# def get_almost_learnt(self):
#     """
#     Rule 2: Find learning item with p_recall above forgot threshold
#     """
#     to_look_at = \
#         self.items[self.learning_progress == self.represent_learning]
#
#     np.random.shuffle(to_look_at)
#
#     for i in to_look_at:
#         if self.p_recall[i] > self.forgot_threshold:
#             return i
#     return None

# def get_useful(self):
#     """
#     Rule 3: Find the most useful item.
#     """
#     to_look_at = \
#         self.items[self.learning_progress != self.represent_learnt]
#
#     np.random.shuffle(to_look_at)
#     max_ind = None
#     max_val = - np.inf
#     for i in to_look_at:
#         if self.usefulness[i] > max_val:
#             if self.taboo != i:
#                 max_val = self.usefulness[i]
#                 max_ind = i
#     if max_ind is None:
#         # print("All items learnt by Learner")
#         # find the least learnt item
#         max_ind = np.argmin(self.p_recall)
#
#     return max_ind
