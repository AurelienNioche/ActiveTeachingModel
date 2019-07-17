import copy
import random
from teacher.metaclass import GenericTeacher
import numpy as np


class LeitnerTeacher(GenericTeacher):

    def __init__(self, n_item=20, t_max=200, grade=1, handle_similarities=True, verbose=False):

        super().__init__(n_item=n_item, t_max=t_max, grade=grade, handle_similarities=handle_similarities,
                         verbose=verbose)
        self.learnt = set()
        self.learning = set()
        self.unseen = set()
        self.buffer_threshold = 15
        # initialise buffer_threshhold number of characters to be in learning set
        for i in range(self.buffer_threshold):
            self.learning.add(i)
        self.count = 0
        self.num_learnt = np.zeros(t_max)
        self.past_recall=np.full((n_item,self.buffer_threshold), -1)
        self.taboo = 1

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
    """
    def update_sets(self, agent, n_items):
        for k in range(n_items):
            if agent.p_recall(k) > self.learn_threshhold:
                self.learned[k] = 2

        if self.count>0:
            if self.learned[self.questions[self.count-1]] == 0:
                self.learned[self.questions[self.count - 1]] = 1

    # calculate usefulness and relative parameters.
    """
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
        print(successes)
        #if successes[-1]
        #self.past_recall[taboo]+=1
        if len(self.learnt) == n_items:
            print("All kanjis learnt")
            return 0
        if self.count != 0:
            if successes[self.count] == True:
                self.past_recall[self.taboo][-1] +
        #for item in self.learning:
        #    self.past_recall
        #change sets according to reply if sum of sizes complete
        """
        self.update_sets(agent, n_items)
        #show all characters once
        if self.check < n_items:
            new_question = self.check
            self.check+=1
            return new_question
        else:
            if self.count==0 or self.count==1 or self.count==3 or self.count==4 or self.count==5:
                for i in range(n_items):
                    if self.learned[i] == 0:
                        new_question = i
                        self.learning += 1
                        return new_question
            elif self.count==2 or self.count==8:
                for i in range(n_items):
                    if self.learned[i] == 1:
                        new_question = i
                        self.countlearnt += 1
                        return new_question
            else:
                for i in range(n_items):
                    if self.learned[i] == 1:
                        new_question = i
                        self.countlearnt += 1
                        return new_question
        """
        #new_question = self.taboo
        self.taboo+=1
        return self.taboo