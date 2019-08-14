# import copy

import numpy as np

from teacher.metaclass import GenericTeacher


class Active(GenericTeacher):

    version = 3

    def __init__(self, n_item=20, t_max=200, grades=(1, ),
                 handle_similarities=True, normalize_similarity=False,
                 learnt_threshold=0.95,
                 verbose=False):

        """
        :param n_item: task attribute
        :param t_max: task attribute
        :param grades: task attribute
        :param handle_similarities: task attribute
        :param normalize_similarity: task attribute
        :param learnt_threshold: p_recall(probability of recall) threshold after
        which an item is learnt.
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

        self.learnt_threshold = learnt_threshold

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
                next_p_recall_j_after_i = agent.p_recall(j)
                usefulness += next_p_recall_j_after_i ** 2
            agent.unlearn()
            self.usefulness[i] = usefulness

        # self.usefulness[:] = 0
        #
        # for i in range(self.tk.n_item):
        #     usefulness = 0
        #     for j in range(self.tk.n_item):
        #         next_p_recall_j = agent.p_recall(j, time_index=self.t+1)
        #         agent.learn(i)
        #         next_p_recall_j_after_i = agent.p_recall(j)
        #         agent.unlearn()
        #
        #         usefulness += (next_p_recall_j_after_i-next_p_recall_j)**2
        #     self.usefulness[i] = usefulness

    def _get_next_node(self, agent=None):
        """
        :param agent: as before.
        :return: integer (index of the question to ask).

        Function implements 3 Rules in order.
        """

        self._update_usefulness(agent)

        if self.t > 0:
            self.taboo = self.question
            self.question = None

        self.question = np.random.choice(
            np.where(self.usefulness == np.max(self.usefulness))[0])

        if self.verbose:
            print(f'Teacher rule: {self.rule}')

        return self.question


class ForceLearning(Active):
    version = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        sum_p_recall = np.zeros(self.tk.n_item)

        current_p_recall = np.zeros(self.tk.n_item)
        for i in range(self.tk.n_item):
            current_p_recall[i] = agent.p_recall(i)

            agent.learn(i)

            p_recall = np.zeros(self.tk.n_item)
            for j in range(self.tk.n_item):
                p_recall[j] = agent.p_recall(j)

            agent.unlearn()
            sum_p_recall[i] = np.sum(np.power(p_recall, 2))
            self.usefulness[i] = np.sum(p_recall > self.learnt_threshold)

        self.usefulness -= current_p_recall > self.learnt_threshold
        if max(self.usefulness) <= 0:
            self.usefulness = sum_p_recall
