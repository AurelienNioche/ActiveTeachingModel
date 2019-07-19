import copy
import random

import numpy as np

from teacher.metaclass import GenericTeacher


class LeitnerTeacher(GenericTeacher):

    def __init__(self, n_item=20, t_max=200, grades=(1, ),
                 handle_similarities=True, fractional_success=0.9,
                 buffer_threshold=15, taboo=None, represent_learnt=2,
                 represent_learning=1, represent_unseen=0,
                 verbose=False):
        """
            :var iteration: current iteration number.
            0 at first iteration.
            :param fractional_success: fraction of successful replies required for teaching
            after which an item is learnt
            :param buffer_threshold: integer value in range(0 to n_items):
                *To prevent bottleneck, maximum number of items that can be taught at a time
            :param taboo: integer value in range(0 to n_items)
                *index of the item shown in previous iteration
           """

        super().__init__(n_item=n_item, t_max=t_max, grades=grades,
                         handle_similarities=handle_similarities,
                         verbose=verbose)
        self.iteration = 0
        self.fractional_success = fractional_success
        self.buffer_threshold = buffer_threshold
        self.taboo = taboo
        self.represent_learnt = represent_learnt
        self.represent_learning = represent_learning
        self.represent_unseen = represent_unseen

        # Initialize buffer_threshold number of characters in learning set
        for i in range(self.buffer_threshold):
            self.learning_progress[i] = 1

        # Matrix to store the past successful attempts of every item
        self.past_successes = np.full((n_item, self.buffer_threshold), -1)

        # At i^th index stores probability of picking i^th item
        self.prob = np.zeros(n_item)

        self.learning_progress = np.zeros(n_item)
        # :param learning_progress: list of size n_items containing:
        #     * param represent_learnt: at i^th index when i^th item is learnt
        #     * param represent_learning: at i^th index when i^th item is seen
        #     at least once but hasn't been learnt
        #     * param represent_unseen: at i^th index when i^th item is unseen

    def ask(self):

        question = self.get_next_node(
            questions=self.questions,
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


    def get_next_node(self, questions, successes, agent, n_items):
        """
            :param questions: list of integers (index of questions). Empty at first iteration
            :param successes: list of booleans (True: success, False: failure) for every question
            :param agent: agent object (RL, ACT-R, ...) that implements at least the following methods:
                * p_recall(item): takes index of a question and gives the probability of recall for the agent in current state
                * learn(item): strengthen the association between a kanji and its meaning
                * unlearn(): cancel the effect of the last call of the learn method
            :param n_items:
                * Number of items included (0 ... n-1)
            :return: integer (index of the question to ask)
        """
        count_learnt = 0
        count_unseen = 0
        count_learning = 0
        for i in range(len(self.learning_progress)):
            if self.learning_progress[i] == 2:
                count_learnt += 1
            elif self.learning_progress[i] == 0:
                count_unseen += 1
            else:
                count_learning += 1

        if count_learnt == n_items:
            print("All kanjis learnt")
            self.iteration += 1
            return 0
        # Update the past recall memory
        if self.iteration != 0:
            result = np.where(self.past_successes[self.taboo] == -1)
            if len(result) > 0 and len(result[0]) > 0:
                self.past_successes[self.taboo, result[0][0]] = \
                    int(successes[self.iteration - 1])
            else:
                for i in range(self.buffer_threshold-1):
                    self.past_successes[self.taboo,i] = self.past_successes[self.taboo,i+1]
                self.past_successes[self.taboo, self.buffer_threshold - 1]= int(successes[self.iteration - 1])
        else:
            #no past memory implies a random question to be shown from learning set
            ran = random.randint(0, self.buffer_threshold)
            self.taboo = ran
            self.iteration += 1
            return int(ran)
        for item in range(n_items):
            #modify the learning set
            if self.learning_progress[item] == 1:
                succ_recalls = np.sum(np.where(self.past_successes[item]>0, 1, 0))
                if succ_recalls >= (self.buffer_threshold*self.fractional_success):
                    #send item from learning set to learnt set
                    self.learning_progress[item] = 2
                    count_learnt += 1
                    count_learning -= 1

                    if count_unseen > 0:
                        # choose any item from unseen set and send to learning set
                        result = np.where(self.learning_progress == 0)
                        if len(result) > 0 and len(result[0]) >0:
                            self.learning_progress[result[0][0]] = 1
                            count_unseen -= 1
                            count_learning += 1
                        else:
                            print( "Error- no unseen char found, error in count_unseen computation")
        #update probabilities of items
        if count_learning >= 2:
            self.prob[self.taboo] = 0
            for item in range(n_items):
                if self.learning_progress[item] >= 1 and item != self.taboo:
                    #get the last successful recalls value
                    succ_recalls = np.sum(
                        np.where(self.past_successes[item] > 0, 1, 0))
                    self.prob[item] = self.buffer_threshold + 1 - succ_recalls

            psum = np.sum(self.prob)
            if psum == 0 :
                assert "error - the total probability is 0"

        else:
            # find the only item in learning set else return all learnt
            result = np.where(self.learning_progress == 1)
            #no check for taboo needed
            if len(result) > 0 and len(result[0]) > 0:
                new_question = int(self.learning_progress[result[0][0]])
                self.taboo = new_question
                self.iteration += 1
                print(new_question)
                return new_question
            else:
                print("all kanjis learnt")
                self.iteration += 1
                return 0

        #find a random integer from 0 to psum
        ran = random.randint(0, psum)

        sum = 0
        for item in range(n_items):
            #map the random integer to an item according to its probabilty of being picked
            if ran <= sum + self.prob[item]:
                new_question = item
                self.taboo = new_question
                self.iteration += 1
                return new_question
            sum += self.prob[item]
        print(ran,sum,psum)
        print("Error none")