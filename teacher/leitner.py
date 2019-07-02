import copy

from teacher.metaclass import GenericTeacher


class LeitnerTeacher(GenericTeacher):

    def __init__(self, n_item=20, t_max=100, grade=1, handle_similarities=True, verbose=False):

        super().__init__(n_item=n_item, t_max=t_max, grade=grade, handle_similarities=handle_similarities,
                         verbose=verbose)
        self.learned = [0]*n_item
        self.learn_threshhold=0.95
        self.forgot_threshhold = 0.85
        sel


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
    # a learnt character is represented by 2 in learned, a being learnt character by 1 . updation is required after every iteration.
    def update_sets(self, agent, n_items):
        for i in range(n_items):
            if agent.p_recall(i)>self.learn_threshhold:
                self.learned[i]=2
            else:
                if len(self.questions)>0:
                    self.learned[self.questions[-1]]=1

    def parameters(self,n_items,agent):
        recall = [0] * n_items
        # recall[i] represents probability of recalling kanji[i] at current instant
        usefullness = [0] * n_items
        recall_next = [[0] * n_items] * n_items  # recall_next[i][j] is prob of recalling i with j learnt.
        relative = [[0] * n_items] * n_items  # relative amount by which knowing j helps i
        for item in range(n_items):
            recall[item] = agent.p_recall(item)
            for item2 in range(n_items):
                if item2 != item:
                    agent.learn(item2)
                    recall_next[item][item2] = agent.p_recall(item)
                    relative[item][item2] = recall_next[item][item2] - recall[item]
                    agent.unlearn()
                usefullness[item2] += relative[item][item2]
        return relative,usefullness

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
        relative,usefullness = self.parameters(n_items,agent)
        #Rule1: dont let a learnt kanji slip out of threshhold
        for i in range(n_items):
            if self.learned[i]==2:
                if agent.p_recall(i)<self.learn_threshhold:
                    if questions[-1]!=i:
                        new_question=i
                        self.update_sets(agent, n_items)
                        return new_question
        #Rule2: Bring an almost learnt Kanji to learnt set.
        for i in range(n_items):
            if self.learned[i]==1:
                if agent.p_recall(i)>self.forgot_threshhold:
                    if questions[-1]!=i:
                        new_question=i
                        self.update_sets(agent, n_items)
                        return new_question
        #Rule3; find the most useful kanji.

        maxind = -1
        maxval = -100
        for i in range(n_items):
            if self.learned[i]<2:
                if usefullness[i] > maxval:
                    maxval = usefullness[i]
                    maxind = i
        new_question = maxind
        self.update_sets(agent, n_items)
        # new_question = random.randint(0, n_items-1)
        return new_question