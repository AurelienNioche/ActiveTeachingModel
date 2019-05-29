import numpy as np

from teacher.tracking_teacher import TrackingTeacher


class RandomTeacher(TrackingTeacher):

    def __init__(self, n_item=20, t_max=100, grade=1, handle_similarities=True, verbose=False):

        super().__init__(n_item=n_item, t_max=t_max, grade=grade, handle_similarities=handle_similarities,
                         verbose=verbose)

    def ask(self):

        question = np.random.randint(self.tk.n_item)

        possible_replies = self.get_possible_replies(question)

        if self.verbose:
            print(f"Question chosen: {self.tk.kanji[question]}; "
                  f"correct answer: {self.tk.meaning[question]}; "
                  f"possible replies: {self.tk.meaning[possible_replies]};")

        return question, possible_replies
