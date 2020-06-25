import numpy as np
from tqdm import tqdm

from learner.learner import Learner


class Leitner:

    def __init__(self, learner, delay_factor=2):
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

        self.learner = learner
        self.delay_factor = delay_factor

        self.n_item = self.learner.n_item

        self.box = np.zeros(self.n_item, dtype=int)
        self.waiting_time = np.zeros(self.n_item, dtype=int)
        self.seen = np.zeros(self.n_item, dtype=bool)

        self.c_iter_session = 0

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

        self.c_iter_session += 1
        if self.c_iter_session >= self.learner.n_iter_per_ss:
            self.waiting_time[:] += self.learner.n_iter_between_ss
            self.c_iter_session = 0

    def _find_due_items(self):
        """
        :return: arr : array that contains the items that are due to be shown
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
        :param seen_due: boolean array (True: should be shown)
        :return: arr: contains the items that have been waiting to be shown the
            most.

        Finds those items with maximum waiting time.
        """

        max_wait = np.max(self.waiting_time[due])
        is_max_wait = self.waiting_time == max_wait
        return is_max_wait

    def _pick_least_box(self, is_max_wait):
        """
        :param max_overdue_items: an integer array that contains the items that
            should be shown.
        :return: items_arr: contains the items that are in the lowest box from
            the max_overdue_items.

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
                if np.sum(unseen) > 0:
                    selection = np.arange(self.n_item)[unseen]
                else:
                    b = self.waiting_time == np.max(self.waiting_time)
                    selection = np.arange(self.n_item)[b]
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

        self.learner.update(item)

        self.taboo = item
        self.seen[item] = True
        self._modify_sets(selected_item=item, success=response)
        self._update_wait_time(selected_item=item)

    def teach(self, n_iter, seed=0):

        np.random.seed(seed)
        h = np.zeros(n_iter, dtype=int)

        for t in tqdm(range(n_iter)):
            item = self.ask()

            r = self.learner.reply(item)
            self.update(item=item, response=r)

            h[t] = item

        return h

    @classmethod
    def run(cls, tk):
        learner = Learner.get(tk)
        teacher = cls(learner=learner)
        return teacher.teach(n_iter=tk.n_iter, seed=tk.seed)
