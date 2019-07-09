import numpy as np
import copy

from teacher.metaclass import GenericTeacher


class RandomTeacher(GenericTeacher):

    def __init__(self, n_item=20, t_max=100, grade=1, seed=123,
                 handle_similarities=True, normalize_similarity=False,
                 verbose=False):

        super().__init__(n_item=n_item, t_max=t_max, grade=grade, seed=seed,
                         normalize_similarity=normalize_similarity,
                         handle_similarities=handle_similarities,
                         verbose=verbose)
        self.mat = np.zeros((n_item, t_max))
        self.count=0

    def ask(self):
        question = self.get_next_node(
            questions=self.questions,
            agent=copy.deepcopy(self.agent),
            n_items=self.tk.n_item
        )

        possible_replies = self.get_possible_replies(question)

        if self.verbose:
            print(f"Question chosen: {self.tk.kanji[question]}; "
                  f"correct answer: {self.tk.meaning[question]}; "
                  f"possible replies: {self.tk.meaning[possible_replies]};")

        return question, possible_replies

    def get_next_node(self, questions, agent, n_items):
        question = np.random.randint(n_items)
        for i in range(n_items):
            if agent.p_recall(i) > 0.95:
                self.mat[i, self.count] = 2
            elif agent.p_recall(i)>0:
                self.mat[i, self.count] = 1
        self.count += 1
        return question

