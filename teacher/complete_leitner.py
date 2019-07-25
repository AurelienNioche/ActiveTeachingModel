import numpy as np

from teacher.metaclass import GenericTeacher

np.random.seed(123)


class TraditionalLeitnerTeacher(GenericTeacher):

    def __init__(self, n_item=20, t_max=200, grades=(1, ),
                 handle_similarities=True,
                 normalize_similarity=False,
                 taboo=None, delay_factor=2, represent_unseen=0,
                 verbose=False):
        """
        :param normalize_similarity: bool. Normalized description of
        semantic and graphic connections between items
        :param taboo: integer value in range(0 to n_items). Index of the item
            shown in previous iteration.
        :param represent_unseen: representation of an unseen item(not shown to
        the learner yet).
        :param verbose: displays each question asked and replies at each
        iteration
        :var self.past_successes: Array to store the past successful attempts
        of every item.
        :var self.learning_progress: list of size n_items containing:
        represent_learnt, represent_learning and represent_unseen
        :var self.pick_probability: Array to store probability
        of picking i^th item at i^th index in
        # current iteration
        """

        super().__init__(n_item=n_item, t_max=t_max, grades=grades,
                         handle_similarities=handle_similarities,
                         normalize_similarity=normalize_similarity,
                         verbose=verbose)

        self.taboo = taboo
        self.represent_unseen = represent_unseen
        self.delay_factor = delay_factor
        self.learning_progress = np.zeros(n_item)
        self.wait_time_arr = np.zeros(n_item)

    def modify_sets(self, successes):
        """
        :param successes: list of booleans (True: success, False: failure) for
            every question
        The boxes will be modified according to the following rules:
            * Move an item to the next box for a successful reply by learner
            * Move an item to the previous box for a failure.
        """
        taboo = self.taboo
        if taboo is not None:
            prev_box = self.learning_progress[taboo]
            if successes[self.t-1]:
                self.learning_progress[taboo] += 1
            else:
                if prev_box > 0:
                    self.learning_progress[taboo] -= 1

    def update_wait_time(self):
        """
        Update the waiting time of every item in wait_time_arr according to the following rules:
            * Taboo will have its wait time changed according to its box.
            * The rest of the item's wait time will be increased by 1
        """
        if self.t > 0:
            for i in range(self.tk.n_item):
                if i != self.taboo:
                    self.wait_time_arr[i] += 1
                else:
                    taboo_box = self.learning_progress[self.taboo]
                    self.wait_time_arr[self.taboo] = -(
                                taboo_box * self.delay_factor)

    def find_due_items(self, n_items):
        """
        :return:
        """
        result = np.where(self.wait_time_arr > 0)
        arr = result[0]
        if len(arr) == 0:
            complete_arr = np.zeros(n_items, dtype=int)
            for i in range(n_items):
                complete_arr[i] = i
            complete_arr = np.delete(complete_arr, self.taboo)
            return complete_arr
        return arr

    def find_due_seen_items(self, due_items):

        seen_due_items = np.full(len(due_items), -1)
        count = 0
        for item in due_items:
            if self.seen[item, self.t]:
                seen_due_items[count] = item
                count += 1

        # find most overdue item
        if count == 0:
            seen_due_items = due_items
            count = len(due_items)
        return seen_due_items, count

    def find_max_waiting(self, items_arr):
        max_wait = float('-inf')
        arr = None
        for i in range(len(items_arr)):
            wait_time_item = self.wait_time_arr[items_arr[i]]
            if max_wait < wait_time_item:
                arr = [items_arr[i]]
                max_wait = wait_time_item
            elif max_wait == wait_time_item:
                arr.append(items_arr[i])
        return arr

    def pick_least_box(self, max_overdue_items):
        min_box = float('inf')
        for item in max_overdue_items:
            box = self.learning_progress[item]
            if box < min_box:
                items_arr = [item]
                min_box = box
            if box == min_box:
                items_arr.append(item)
        return items_arr

    def _get_next_node(self, agent=None):
        """
        :param agent: agent object (RL, ACT-R, ...) that implements at least
            the following methods:
            * p_recall(item): takes index of a question and gives the
                probability of recall for the agent in current state
            * learn(item): strengthen the association between a kanji and its
                meaning
            * unlearn(): cancel the effect of the last call of the learn method

        :return: integer (index of the question to ask)
        Function implements 4 rules in order:

        """
        if self.t == 0:
            # No past memory, so a random question shown from learning set
            random_question = np.random.randint(0, self.tk.n_item)
            self.taboo = random_question
            return int(random_question)
        successes = self.successes[:self.t]
        self.modify_sets(successes)
        self.update_wait_time()

        # Criteria 1, get due items
        due_items = self.find_due_items(self.tk.n_item)

        # preference for seen item
        seen_due_items, count = self.find_due_seen_items(due_items)

        # items with maximum waiting time
        max_overdue_items = self.find_max_waiting(seen_due_items[:count])

        # pick item in lowest box
        least_box_items = self.pick_least_box(max_overdue_items)

        if len(least_box_items) > 1:
            pick_question_index = np.random.randint(len(least_box_items))
            new_question = least_box_items[pick_question_index]
        else:
            new_question = least_box_items[0]

        self.taboo = new_question
        return new_question
