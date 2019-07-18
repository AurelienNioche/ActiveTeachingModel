import copy
import random
from teacher.metaclass import GenericTeacher
import numpy as np

#
class LeitnerTeacher(GenericTeacher):

    def __init__(self, n_item=20, t_max=200, grade=1, handle_similarities=True, verbose=False):

        super().__init__(n_item=n_item, t_max=t_max, grade=grade, handle_similarities=handle_similarities,
                         verbose=verbose)
        self.teach_set = np.zeros(n_item)
        self.buffer_threshold = 15
        # initialise buffer_threshhold number of characters to be in learning set
        for i in range(self.buffer_threshold):
            self.teach_set[i] = 1
        self.count = 0
        self.num_learnt = np.zeros(t_max)
        self.past_successes = np.full((n_item, self.buffer_threshold), -1)
        self.taboo = -1
        self.prob = np.zeros(n_item)
        self.prob_threshhold = 0.9

    def ask(self):

        question = self.get_next_node(
            questions=self.questions,
            successes=self.successes,
            agent=copy.deepcopy(self.agent),
            n_items=self.tk.n_item
        )
        possible_replies = self.get_possible_replies(question)

        if self.verbose:
            print(f"Question chosen: {self.tk.kanji[question]}; "
                  f"correct answer: {self.tk.meaning[question]}; "
                  f"possible replies: {self.tk.meaning[possible_replies]};")

        return question, possible_replies

    # Learnt character set is represented by 2,Being learnt character set by 1, Unseen character set as 0

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
        for i in range(len(self.teach_set)):
            if self.teach_set[i] == 2:
                count_learnt += 1
            elif self.teach_set[i] == 0:
                count_unseen += 1
            else:
                count_learning += 1
        self.num_learnt[self.count]=count_learnt
        if count_learnt == n_items:
            print("All kanjis learnt")
            self.count += 1
            return 0
        #update the past recall memory
        if self.count != 0:
            result = np.where(self.past_successes[self.taboo] == -1)
            if len(result) > 0 and len(result[0]) > 0:
                self.past_successes[self.taboo, result[0][0]] = int(successes[self.count-1])
            else:
                for i in range(self.buffer_threshold-1):
                    self.past_successes[self.taboo,i] = self.past_successes[self.taboo,i+1]
                self.past_successes[self.taboo, self.buffer_threshold - 1]= int(successes[self.count-1])
        else:
            ran = random.randint(0, n_items)
            self.taboo = ran
            self.count += 1
            return ran
        for item in range(n_items):
            if self.teach_set[item] == 1:
                succ_recalls = np.sum(np.where(self.past_successes[item]>0, 1, 0))

                if succ_recalls >= (self.buffer_threshold*self.prob_threshhold):
                    self.teach_set[item] = 2
                    count_learnt += 1
                    count_learning -= 1

                    if count_unseen > 0:
                        # choose any item from unseen
                        result = np.where(self.teach_set == 0)
                        if len(result) > 0 and len(result[0]) >0:
                            self.teach_set[result[0][0]] = 1
                            count_unseen -= 1
                            count_learning += 1
                        else:
                            print( "error- no unseen char found, error in count_unseen computation")

        if count_learning >= 2:
            self.prob[self.taboo] = 0
            for item in range(n_items):
                if self.teach_set[item] >= 1 and item != self.taboo:
                    #get the last successful recalls value
                    succ_recalls = np.sum(
                        np.where(self.past_successes[item] > 0, 1, 0))
                    self.prob[item] = self.buffer_threshold + 1 - succ_recalls

            psum = np.sum(self.prob)
            if psum == 0 :
                assert "error - the total probability is 0"

        else:
            # find the only element or if no element then return learnt
            result = np.where(self.teach_set == 1)
            #no check for taboo needed
            if len(result) > 0 and len(result[0]) > 0:
                new_question = self.teach_set[result[0][0]]
                self.taboo = new_question
                self.count += 1
                return new_question
            else:
                print("all kanjis learnt")
                self.count += 1
                return 0


        #find random from 0 to psum
        ran = random.randint(0, psum)

        sum = 0
        for item in range(n_items):
            #print(sum)
            if ran <= sum + self.prob[item]:
                new_question = item
                self.taboo = new_question
                self.count += 1
                return new_question
            sum += self.prob[item]
        print(ran,sum,psum)
        print("Error none")