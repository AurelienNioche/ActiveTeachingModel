# import copy

import numpy as np

from teacher.metaclass import GenericTeacher


class AvyaTeacher(GenericTeacher):

    represent_unseen = 0
    represent_learning = 1
    represent_learnt = 2

    def __init__(self, n_item=20, t_max=200, grades=(1, ),
                 handle_similarities=True, normalize_similarity=False,
                 learnt_threshold=0.95,
                 # forgot_threshold=0.85,
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
        # self.forgot_threshold = forgot_threshold

        self.seen_items = np.zeros(n_item, dtype=bool)

        self.p_recall = np.zeros(self.tk.n_item)
        self.usefulness = np.zeros(self.tk.n_item)

        self.items = np.arange(self.tk.n_item)

        self.question = None
        self.taboo = None
        self.not_taboo = np.ones(self.tk.n_item, dtype=bool)

        self.rule = None

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v, ord=1)
        if norm == 0:
            return v
        return v / norm

    def _update_usefulness_and_p_recall(self, agent):
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
            self.p_recall[i] = agent.p_recall(i)
            p_recall_i_next_it = agent.p_recall(i, time_index=self.t+1)
            for j in range(self.tk.n_item):
                if j != i:
                    agent.learn(j)
                    p_recall_i_after_j = agent.p_recall(i)
                    agent.unlearn()

                    relative = p_recall_i_after_j - p_recall_i_next_it
                    self.usefulness[j] += relative

    @staticmethod
    def _p_recall_influence(x):

        # return 1/(1+np.exp((x-0.98)/0.005))
        return 1/(1+np.exp((x-0.7)/0.1))

    def check_for_new_item(self):

        if np.sum(self.seen_items)\
                == np.sum(self.p_recall > self.learn_threshold) or \
                (np.sum(self.seen_items) == 1 and
                 self.seen_items[self.taboo] == 1):
            selection = self.items[self.seen_items == 0]
            if len(selection):
                np.random.shuffle(selection)
                self.question = \
                    selection[np.argmax(self.usefulness[selection])]
                self.rule = 'New'

    def pickup_already_seen(self):

        self.rule = 'Already seen'

        already_seen = self.seen_items == 1
        selection = self.items[already_seen * self.not_taboo]
        if len(selection) == 1:
            self.question = self.items[selection[0]]
            return

        p_recall_inf = self._p_recall_influence(self.p_recall[selection])
        norm_p_recall_inf = self.normalize(p_recall_inf)
        norm_useful = self.normalize(self.usefulness[selection])
        v = norm_p_recall_inf + norm_useful
        p = v / np.sum(v)

        if self.verbose:
            print("p recall", self.p_recall[selection])
            print('p recall influence', p_recall_inf)
            print('p recall influence NORM', norm_p_recall_inf)
            print('usefulness', self.usefulness[selection])
            print('usefulness NORM', norm_useful)
            print('p', p)

        self.question = np.random.choice(self.items[selection], p=p)

    def _get_next_node(self, agent=None):
        """
        :param agent: as before.
        :return: integer (index of the question to ask).

        Function implements 3 Rules in order.
        """

        # agent = copy.deepcopy(agent)
        self._update_usefulness_and_p_recall(agent)

        if self.t > 0:
            self.taboo = self.question
            self.question = None
            self.not_taboo[:] = self.items != self.taboo

        self.check_for_new_item()
        if self.question is None:
            self.pickup_already_seen()

        self.seen_items[self.question] = True

        if self.verbose:
            print(f'Teacher rule: {self.rule}')

        return self.question
