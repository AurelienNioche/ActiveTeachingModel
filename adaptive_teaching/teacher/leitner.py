import numpy as np

from . generic import GenericTeacher


class Leitner(GenericTeacher):

    def __init__(self, delay_factor=2, **kwargs):
        """
        :param normalize_similarity: bool.
            Normalized description of
            semantic and graphic connections between items
        :param delay_factor:
        :param verbose:
            displays each question asked and replies at each
            iteration
        :var self.taboo:
            integer value in range(0 to n_items). Index of the item
            shown in previous iteration.
        :var self.box:
            array of size n_item representing the box
            number of i^th item at i^th index.
        :var self.wait_time_arr:
            array of size n_item representing the waiting time
            of i^th item at i^th index.
        """

        super().__init__(**kwargs)

        self.delay_factor = delay_factor
        self.box = np.zeros(self.n_item)
        self.wait_time_arr = np.zeros(self.n_item)

        self.taboo = None

        # Time step (integer)
        self.t = 0

        # Historic of success (bool)
        self.hist_success = []

        # Historic of presented items (int)
        self.hist_item = []

    def _modify_sets(self):
        """
        The boxes will be modified according to the following rules:
            * Move an item to the next box for a successful reply by learner
            * Move an item to the previous box for a failure.
        """
        taboo = self.taboo
        prev_box = self.box[taboo]
        if self.hist_success[self.t-1]:
            self.box[taboo] += 1
        else:
            if prev_box > 0:
                self.box[taboo] -= 1

    def _update_wait_time(self):
        """
        Update the waiting time of every item in wait_time_arr according to the
         following rules:
            * Taboo will have its wait time changed according to its box.
            * The rest of the item's wait time will be increased by 1
        """

        for i in range(self.n_item):
            if i != self.taboo:
                self.wait_time_arr[i] += 1
            else:
                self.wait_time_arr[self.taboo] =\
                    - self.delay_factor**self.box[self.taboo]

    def _find_due_items(self):
        """
        :return: arr : array that contains the items that are due to be shown
        Pick all items with positive waiting time.

        Suppose there exist no due item then pick all items except taboo.
        """
        result = np.where(self.wait_time_arr > 0)
        arr = result[0]
        if len(arr) == 0:
            complete_arr = np.arange(self.n_item)
            arr = np.delete(complete_arr, self.taboo)
        return arr

    def _find_due_seen_items(self, due_items):
        """
        :param due_items: array that contains items that are due to be shown
        :return: * seen_due_items: as before
                * count: the count of the number of items in seen__due_items

        Finds the items that are seen and are due to be shown.
        """

        # integer array with size of due_items
        #                 * Contains the items that have been seen
        #                     at least once and are due to be shown.
        seen_due_items = np.intersect1d(self.hist_item, due_items)
        count = len(seen_due_items)

        if count == 0:
            seen_due_items = due_items
            count = len(due_items)
        return seen_due_items, count

    def _find_max_waiting(self, items_arr):
        """
        :param items_arr: an integer array that contains the items that should
                be shown.
        :return: arr: contains the items that have been waiting to be shown the
            most.

        Finds those items with maximum waiting time.
        """

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

    def _pick_least_box(self, max_overdue_items):
        """
        :param max_overdue_items: an integer array that contains the items that
            should be shown.
        :return: items_arr: contains the items that are in the lowest box from
            the max_overdue_items.

        Finds the items present in the lowest box number.
        """

        items_arr = []
        min_box = float('inf')
        for item in max_overdue_items:
            box = self.box[item]
            if box < min_box:
                items_arr = [item]
                min_box = box
            elif box == min_box:
                items_arr.append(item)
        assert len(items_arr), 'This should not be empty'

        return items_arr

    def _get_next_node(self):
        """
        :return: integer (index of the question to ask)

        Every item is associated with:
            * A waiting time i.e the time since it was last shown
            to the learner.
                -- maintained in variable wait_time_arr
            * A box that decides the frequency of repeating an item.
                -- maintained in variable learning_progress
        Function implements 4 rules in order:
            1. The items that are due to be shown are picked.
            2. Preference given to the seen items.
            3. Preference given to the items present in lowest box number.
            4. Randomly pick from the said items.

        """
        if self.t == 0:
            # No past memory, so a random question shown from learning set
            random_question = np.random.randint(0, self.n_item)
            self.taboo = random_question
            return int(random_question)

        self._modify_sets()
        self._update_wait_time()

        # Criteria 1, get due items
        due_items = self._find_due_items()

        # preference for seen item
        seen_due_items, count = self._find_due_seen_items(due_items=due_items)

        # items with maximum waiting time
        max_overdue_items = self._find_max_waiting(seen_due_items[:count])

        # pick item in lowest box
        least_box_items = self._pick_least_box(max_overdue_items)
        new_question = np.random.choice(least_box_items)

        self.taboo = new_question
        return new_question

    def ask(self, best_param):

        return self._get_next_node()

    def update(self, item, response):

        self.hist_success.append(response)
        self.hist_item.append(item)
        self.t += 1
