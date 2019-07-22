import copy
import random

import numpy as np

from teacher.metaclass import GenericTeacher


class LeitnerTeacher(GenericTeacher):

    def __init__(self, n_item=20, t_max=200, grades=(1, ), seed=123,
                 handle_similarities=True, fractional_success=0.9,
                 buffer_threshold=15, taboo=None, represent_learnt=2,
                 represent_learning=1, represent_unseen=0,
                 verbose=False):
        """
        :var iteration: current iteration number.
        0 at first iteration.
        :param fractional_success: float fraction of successful replies
            required for teaching after which an item is learnt.
        :param buffer_threshold: integer value in range(0 to n_items):
            * To prevent bottleneck, maximum number of items that can be
            taught at a time
        :param taboo: integer value in range(0 to n_items). Index of the item
            shown in previous iteration.
        :param represent_learnt: representation of a learnt item(Learnt by
        learner).
        :param represent_learning: representation of a learning item(shown
        atleast once but has not been learnt by learner).
        :param represent_unseen: representation of an unseen item(not shown to
        the learner yet).
        :param verbose: displays each question asked and replies at each
        iteration
        :var past_successes: Array to store the past successful attempts of
        every item.
        :var learning_progress: list of size n_items containing:
        represent_learnt, represent_learning and represent_unseen
        :var pick_probability: Array to store probability of picking i^th item
        at i^th index in
        # current iteration
        """

        super().__init__(n_item=n_item, t_max=t_max, grades=grades, seed=seed,
                         handle_similarities=handle_similarities,
                         verbose=verbose)
        self.iteration = 0
        self.fractional_success = fractional_success
        self.buffer_threshold = buffer_threshold
        self.taboo = taboo
        self.represent_learnt = represent_learnt
        self.represent_learning = represent_learning
        self.represent_unseen = represent_unseen

        self.past_successes = np.full((n_item, self.buffer_threshold), -1)
        self.learning_progress = np.zeros(n_item)
        self.pick_probability = np.zeros(n_item)

        # Initialize certain number of characters in learning set according to
        # buffer_threshold to prevent bottleneck.
        for i in range(self.buffer_threshold):
            self.learning_progress[i] = self.represent_learning

    def ask(self):

        question = self.get_next_node(
            successes=self.successes,
            agent=copy.deepcopy(self.agent),
            n_items=self.tk.n_item
        )
        print(question)
        possible_replies = self.get_possible_replies(question)

        if self.verbose:
            print(f"Question chosen: {self.tk.kanji[question]}; "
                  f"correct answer: {self.tk.meaning[question]}; "
                  f"possible replies: {self.tk.meaning[possible_replies]};")

        return question, possible_replies

    def update_probabilities(self, n_items):
        """
        :param n_items: n = Number of items included (0 ... n-1)
        :var prob: array that stores probability of picking i^th item at i^th
        index

        Updates the pick_probability array at every iteration. The probability
        of picking taboo is 0. For rest, the probability of an item(P) =
        buffer_threshold(int value) + 1 - number of successes
        of that item in past buffer_threshold(int value) iterations of that
        item.
        """
        self.pick_probability[self.taboo] = 0
        for item in range(n_items):
            if self.learning_progress[item] >= 1 and item != self.taboo:
                successful_recalls = np.sum(
                    np.where(self.past_successes[item] > 0, 1, 0))
                self.pick_probability[item] = self.buffer_threshold +\
                                              1 - successful_recalls

    def store_item_past(self, successes):
        """
        :param successes: list of booleans (True: success, False: failure)
            for every question.

        Updates the last shown item's past.
        """
        result = np.where(self.past_successes[self.taboo] == -1)
        if len(result) > 0 and len(result[0]) > 0:
            self.past_successes[self.taboo, result[0][0]] = \
                int(successes[self.iteration - 1])
        else:
            for i in range(self.buffer_threshold - 1):
                self.past_successes[self.taboo, i] = self.past_successes[
                    self.taboo, i + 1]
            self.past_successes[self.taboo, self.buffer_threshold - 1] = int(
                successes[self.iteration - 1])

    def modify_sets(self, n_items, count_learning, count_learnt, count_unseen):
        """
        :param n_items: int number of items included from 0 to n-1 characters.
        :param count_learning: int number of items that are represented by
        represent_learning.
        :param count_learnt: int number of items that are represented by
        represent_learnt.
        :param count_unseen: int number of items that have not been shown.
        :return: int number of items in modified Learning set.

        Shifts elements in between the following sets: Unseen, Learning and
        Learnt according to the following Rule:
            * Move an item from learning set to learnt when the number of past
            successful attempts for that item exceeds
            buffer_threshold * fractional_success. Also move an unseen
            character from unseen set to learning set, to fill up the vacancy.
        """
        for item in range(n_items):
            if self.learning_progress[item] == self.represent_learning:
                count_successes = np.sum(np.where(self.past_successes[item] >
                                                  0, 1, 0))
                if count_successes >= (self.buffer_threshold
                                       * self.fractional_success):
                    self.learning_progress[item] = self.represent_learnt
                    count_learnt += 1
                    count_learning -= 1

                    if count_unseen > 0:
                        result = np.where(self.learning_progress ==
                                          self.represent_unseen)
                        if len(result) > 0 and len(result[0]) > 0:
                            self.learning_progress[result[0][0]] = \
                                self.represent_learning
                            count_unseen -= 1
                            count_learning += 1
                        else:
                            print("Error: no unseen char found")
        return count_learning

    def get_next_node(self, successes, agent, n_items):
        """
        :param questions: list of integers (index of questions). Empty at first
            iteration
        :param successes: list of booleans (True: success, False: failure) for
            every question
        :param agent: agent object (RL, ACT-R, ...) that implements at least
            the following methods:
            * p_recall(item): takes index of a question and gives the
                probability of recall for the agent in current state
            * learn(item): strengthen the association between a kanji and its
                meaning
            * unlearn(): cancel the effect of the last call of the learn method
        :param n_items:
            * Number of items included (0 ... n-1)
        :return: integer (index of the question to ask)
        """

        # Items in each set
        count_learnt = 0
        count_unseen = 0
        count_learning = 0

        for i in range(len(self.learning_progress)):
            if self.learning_progress[i] == self.represent_learnt:
                count_learnt += 1
            elif self.learning_progress[i] == self.represent_unseen:
                count_unseen += 1
            else:
                count_learning += 1

        if count_learnt == n_items:
            print("All items learnt by Learner")
            # find the least learnt item
            recall_arr = np.zeros(n_items)
            for i in range(n_items):
                recall_arr[i] = agent.p_recall(i)
            result = np.where(recall_arr == np.amin(recall_arr))
            new_question = result[0][0]
            self.iteration += 1
            return new_question

        if self.iteration == 0:
            # No past memory, so a random question shown from learning set
            random_question = random.randint(0, self.buffer_threshold)
            self.taboo = random_question
            self.iteration += 1
            return int(random_question)
        else:
            self.store_item_past(successes)

        count_learning = self.modify_sets(n_items, count_learning,
                                          count_learnt, count_unseen)

        # Update probabilities of items
        if count_learning >= 2:
            self.update_probabilities(n_items)
        else:
            # Ask the item left in learning set
            result = np.where(self.learning_progress ==
                              self.represent_learning)
            if len(result) > 0 and len(result[0]) > 0:
                new_question = int(self.learning_progress[result[0][0]])
                if self.taboo != new_question:
                    self.taboo = new_question
                    self.iteration += 1
                    return new_question
                else:
                    self.update_probabilities(n_items)
            else:
                print("All items learnt by Learner")
                recall_arr = np.zeros(n_items)
                for i in range(n_items):
                    recall_arr[i] = agent.p_recall(i)
                result = np.where(recall_arr == np.amin(recall_arr))
                new_question = result[0][0]
                self.iteration += 1
                return new_question

        probability_sum = np.sum(self.pick_probability)
        assert(probability_sum)
        pick_question = random.randint(0, probability_sum)  # TODO do you have a seed?

        iterating_sum = 0
        for item in range(n_items):
            # Choose an item according to its probability of being picked
            if pick_question <= iterating_sum + self.pick_probability[item]:
                new_question = item
                self.taboo = new_question
                self.iteration += 1
                return new_question
            iterating_sum += self.pick_probability[item]
        print("Error -- no question returned")  # TODO do the error thingy thing
