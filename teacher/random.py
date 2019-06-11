import numpy as np

from teacher.metaclass import GenericTeacher


class RandomTeacher(GenericTeacher):

    def __init__(self, n_item=20, t_max=100, grade=1, seed=123, handle_similarities=True, normalize_similarity=False,
                 verbose=False):

        super().__init__(n_item=n_item, t_max=t_max, grade=grade, seed=seed,
                         normalize_similarity=normalize_similarity,
                         handle_similarities=handle_similarities,
                         verbose=verbose)

    def ask(self):

        question = np.random.randint(self.tk.n_item)

        possible_replies = self.get_possible_replies(question)

        if self.verbose:
            print(f"Question chosen: {self.tk.kanji[question]}; "
                  f"correct answer: {self.tk.meaning[question]}; "
                  f"possible replies: {self.tk.meaning[possible_replies]};")

        return question, possible_replies