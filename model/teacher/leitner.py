import numpy as np


class Leitner:

    def __init__(self, n_item, delay_factor=2):
        """
        :param delay_factor:
        :var self.taboo:
            integer value in range(0 to n_items). Index of the item
            shown in previous iteration.
        :var self.box:
            array of size n_item representing the box
            number of i^th item at i^th index.
        :var self.waiting_time:
            array of size n_item representing the waiting time
            of i^th item at i^th index.
        """

        self.n_item = n_item

        self.delay_factor = delay_factor
        self.box = np.zeros(self.n_item, dtype=int)
        self.waiting_time = np.zeros(self.n_item, dtype=int)
        self.seen = np.zeros(self.n_item, dtype=bool)

        self.taboo = None

    def _modify_sets(self, selected_item, success):
        """
        The boxes will be modified according to the following rules:
            * Move an item to the next box for a successful reply by learner
            * Move an item to the previous box for a failure.
        """
        prev_box = self.box[selected_item]
        if success:
            self.box[selected_item] += 1
        else:
            if prev_box > 0:
                self.box[selected_item] -= 1

    def _update_wait_time(self, selected_item):
        """
        Update the waiting time of every item in waiting_time according to the
         following rules:
            * Taboo will have its wait time changed according to its box.
            * The rest of the item's wait time will be increased by 1
        """

        self.waiting_time[:] += 1
        self.waiting_time[selected_item] =\
            - self.delay_factor**self.box[selected_item]

    def _find_due_items(self):
        """
        :return: due : boolean array (True: due to be shown)
        Pick all seen items with positive waiting time, except taboo.
        """

        # get due items
        due = self.waiting_time >= 0
        due[self.taboo] = False

        # preference for seen item
        due *= self.seen

        return due

    def _find_max_waiting(self, due):
        """
        :param due: boolean array (True: should be shown)
        :return: is_max_wait: boolean array
        (True: have been waiting to be shown the most).

        Finds those items with maximum waiting time.
        """

        max_wait = np.max(self.waiting_time[due])
        is_max_wait = self.waiting_time == max_wait
        return is_max_wait

    def _pick_least_box(self, is_max_wait):
        """
        :param is_max_wait: boolean array
        :return: finally_due: boolean array
        (True: are in the lowest box and
        have been waiting to be shown the most).

        Finds the items present in the lowest box number.
        """

        min_box = np.min(self.box[is_max_wait])
        is_min_box = self.box == min_box
        finally_due = is_min_box * is_max_wait
        return finally_due

    def ask(self):
        """
        :return: integer (index of the question to ask)

        Every item is associated with:
            * A waiting time i.e the time since it was last shown
            to the learner.
                -- maintained in variable waiting_time
            * A box that decides the frequency of repeating an item.
                -- maintained in variable learning_progress
        Function implements 4 rules in order:
            1. The items that are due to be shown are picked.
            2. Preference given to the seen items.
            3. Preference given to the items present in lowest box number.
            4. Randomly pick from the said items.

        """
        if self.taboo is None:
            # No past memory, so a random question shown from learning set
            selection = np.arange(self.n_item)
        else:
            # get due items
            due = self._find_due_items()

            if np.sum(due) == 0:
                # Present a new item
                unseen = np.logical_not(self.seen)
                selection = np.arange(self.n_item)[unseen]

            else:
                # select maximum waiting time
                max_wait = self._find_max_waiting(due)

                # among max waiting time, find item in lowest box
                least_box = self._pick_least_box(max_wait)
                selection = np.arange(self.n_item)[least_box]

        new_question = np.random.choice(selection)
        self.taboo = new_question
        return new_question

    def update(self, item, response):

        self.taboo = item
        self.seen[item] = True
        self._modify_sets(selected_item=item, success=response)
        self._update_wait_time(selected_item=item)


# import numpy as np
#
#
# class Leitner:
#
#     def __init__(self, n_item, delay_factor=2):
#         """
#         :param delay_factor:
#         :var self.taboo:
#             integer value in range(0 to n_items). Index of the item
#             shown in previous iteration.
#         :var self.box:
#             array of size n_item representing the box
#             number of i^th item at i^th index.
#         :var self.waiting_time:
#             array of size n_item representing the waiting time
#             of i^th item at i^th index.
#         """
#
#         self.n_item = n_item
#
#         self.delay_factor = delay_factor
#         self.box = np.zeros(self.n_item)
#         self.wait_time_arr = np.zeros(self.n_item)
#
#         self.taboo = None
#
#         # Time step (integer)
#         self.t = 0
#
#         # Historic of success (bool)
#         self.hist_success = []
#
#         # Historic of presented items (int)
#         self.hist_item = []
#
#     def _modify_sets(self):
#         """
#         The boxes will be modified according to the following rules:
#             * Move an item to the next box for a successful reply by learner
#             * Move an item to the previous box for a failure.
#         """
#         taboo = self.taboo
#         prev_box = self.box[taboo]
#         if self.hist_success[self.t-1]:
#             self.box[taboo] += 1
#         else:
#             if prev_box > 0:
#                 self.box[taboo] -= 1
#
#     def _update_wait_time(self):
#         """
#         Update the waiting time of every item in waiting_time according to the
#          following rules:
#             * Taboo will have its wait time changed according to its box.
#             * The rest of the item's wait time will be increased by 1
#         """
#
#         for i in range(self.n_item):
#             if i != self.taboo:
#                 self.wait_time_arr[i] += 1
#             else:
#                 self.wait_time_arr[self.taboo] =\
#                     - self.delay_factor**self.box[self.taboo]
#
#     def _find_due_items(self):
#         """
#         :return: arr : array that contains the items that are due to be shown
#         Pick all items with positive waiting time.
#
#         Suppose there exist no due item then pick all items except taboo.
#         """
#         result = np.where(self.wait_time_arr > 0)[0]
#         if len(result) == 0:
#             result = np.delete(np.arange(self.n_item), self.taboo)
#         return result
#
#     def _find_due_seen_items(self, due_items):
#         """
#         :param due_items: array that contains items that are due to be shown
#         :return: * seen_due_items: as before
#                 * count: the count of the number of items in seen__due_items
#
#         Finds the items that are seen and are due to be shown.
#         """
#
#         # integer array with size of due_items
#         #                 * Contains the items that have been seen
#         #                     at least once and are due to be shown.
#         seen_due_items = np.intersect1d(self.hist_item, due_items)
#         count = len(seen_due_items)
#
#         if count == 0:
#             seen_due_items = due_items
#         return seen_due_items
#
#     def _find_max_waiting(self, items_arr):
#         """
#         :param items_arr: an integer array that contains the items that should
#                 be shown.
#         :return: arr: contains the items that have been waiting to be shown the
#             most.
#
#         Finds those items with maximum waiting time.
#         """
#
#         max_wait = float('-inf')
#         arr = None
#         for i in range(len(items_arr)):
#             wait_time_item = self.wait_time_arr[items_arr[i]]
#             if max_wait < wait_time_item:
#                 arr = [items_arr[i]]
#                 max_wait = wait_time_item
#             elif max_wait == wait_time_item:
#                 arr.append(items_arr[i])
#         return arr
#
#     def _pick_least_box(self, max_overdue_items):
#         """
#         :param max_overdue_items: an integer array that contains the items that
#             should be shown.
#         :return: items_arr: contains the items that are in the lowest box from
#             the max_overdue_items.
#
#         Finds the items present in the lowest box number.
#         """
#
#         items_arr = []
#         min_box = float('inf')
#         for item in max_overdue_items:
#             box = self.box[item]
#             if box < min_box:
#                 items_arr = [item]
#                 min_box = box
#             elif box == min_box:
#                 items_arr.append(item)
#         assert len(items_arr), 'This should not be empty'
#
#         return items_arr
#
#     def _get_next_node(self):
#         """
#         :return: integer (index of the question to ask)
#
#         Every item is associated with:
#             * A waiting time i.e the time since it was last shown
#             to the learner.
#                 -- maintained in variable waiting_time
#             * A box that decides the frequency of repeating an item.
#                 -- maintained in variable learning_progress
#         Function implements 4 rules in order:
#             1. The items that are due to be shown are picked.
#             2. Preference given to the seen items.
#             3. Preference given to the items present in lowest box number.
#             4. Randomly pick from the said items.
#
#         """
#         if self.t == 0:
#             # No past memory, so a random question shown from learning set
#             random_question = np.random.randint(0, self.n_item)
#             self.taboo = random_question
#             return int(random_question)
#
#         self._modify_sets()
#         self._update_wait_time()
#
#         # Criteria 1, get due items
#         due_items = self._find_due_items()
#
#         # preference for seen item
#         seen_due_items = self._find_due_seen_items(due_items=due_items)
#
#         # items with maximum waiting time
#         max_overdue_items = self._find_max_waiting(seen_due_items)
#
#         # pick item in lowest box
#         least_box_items = self._pick_least_box(max_overdue_items)
#         new_question = np.random.choice(least_box_items)
#
#         self.taboo = new_question
#         return new_question
#
#     def ask(self):
#         return self._get_next_node()
#
#     def update(self, item, response):
#
#         self.hist_success.append(response)
#         self.hist_item.append(item)
#         self.t += 1
