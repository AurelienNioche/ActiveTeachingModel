import copy

from teacher.tracking_teacher import TrackingTeacher

from solver import solver


class NirajTeacher(TrackingTeacher):

    def __init__(self, n_item=20, t_max=100, grade=1, handle_similarities=True, verbose=False):

        super().__init__(n_item=n_item, t_max=t_max, grade=grade, handle_similarities=handle_similarities,
                         verbose=verbose)

    def ask(self):

        question = solver.get_next_node(
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
