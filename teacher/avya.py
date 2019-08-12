import copy

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

        self.usefulness = np.zeros(self.tk.n_item)

        self.items = np.arange(self.tk.n_item)

        self.question = None
        self.taboo = None

        self.rule = None

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v, ord=1)
        if norm == 0:
            return v
        return v / norm

    def _update_usefulness(self, agent):
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
            usefulness = 0
            agent.learn(i)
            for j in range(self.tk.n_item):
                usefulness += agent.p_recall(j)**2
            agent.unlearn()
            self.usefulness[i] = usefulness

    def _get_next_node(self, agent=None):
        """
        :param agent: as before.
        :return: integer (index of the question to ask).

        Function implements 3 Rules in order.
        """

        # agent = copy.deepcopy(agent)
        self._update_usefulness(agent)

        if self.t > 0:
            self.taboo = self.question
            self.question = None

        self.question = np.argmax(self.usefulness)

        if self.verbose:
            print(f'Teacher rule: {self.rule}')

        return self.question
