import numpy as np

from task.models import User, Question, Kanji
from task.parameters import N_POSSIBLE_REPLIES

import similarity_graphic.measure
import similarity_semantic.measure

from . data_structure import Task, Data


def show_in_console():

    users = User.objects.all().order_by('id')

    for user in users:

        user_id = user.id

        print(f"User {user_id}")
        print("*" * 4)

        que = Question.objects.filter(user_id=user_id).order_by('t')

        for q in que:
            t = q.t
            reaction_time = int((q.time_reply - q.time_display).total_seconds() * 10 ** 3)
            success = q.reply == q.correct_answer

            to_print = f't:{t}, question: {q.question}, reply: {q.reply}, ' \
                f'correct answer: {q.correct_answer}, ' \
                f'success: {"Yes" if success else "No"}, ' \
                f'reaction time: {reaction_time} ms'
            print(to_print)

        print()


class UserTask(Task):

    def __init__(self, user_id,
                 normalize_similarity=False,
                 compute_similarity=True,
                 verbose=False, ):

        super().__init__(t_max=self.question_entries[-1].t,
                         n_possible_replies=N_POSSIBLE_REPLIES)

        self.question_entries = \
            [q for q in Question.objects.filter(user_id=user_id).order_by('t')]

        answers = []

        for q in self.question_entries:

            for i in range(N_POSSIBLE_REPLIES):
                answer = getattr(q, f'possible_reply_{i}')
                answers.append(answer)

        u_answers = np.unique(answers)

        self.meaning = list(u_answers)
        self.meaning.sort()
        self.kanji = []

        for m in self.meaning:
            try:
                k = Question.objects.filter(correct_answer=m)[0].question

            except IndexError:
                k = Kanji.objects.filter(meaning=m).order_by('grade')[0].kanji

            self.kanji.append(k)

        self.n_item = len(self.kanji)

        if compute_similarity:
            self.c_graphic = similarity_graphic.measure.get(
                self.kanji, normalize_similarity=normalize_similarity)
            self.c_semantic = similarity_semantic.measure.get(
                self.meaning, normalize_similarity=normalize_similarity)

        if verbose:
            print(f"Kanji used: {self.kanji}")
            print(f"Corresponding meanings: {self.meaning}\n")


class UserData:

    def __init__(self, user_id, normalize_similarity=False, verbose=False):

        self.tk = UserTask(user_id=user_id,
                           normalize_similarity=normalize_similarity,
                           verbose=verbose)

        self.n_items = self.tk.n_item
        t_max = self.tk.t_max

        self.questions = np.zeros(t_max, dtype=int)
        self.replies = np.zeros(t_max, dtype=int)
        self.possible_replies = np.zeros((t_max, self.tk.n_possible_replies),
                                         dtype=int)
        self.success = np.zeros(t_max, dtype=int)

        for t in range(t_max):

            self.questions[t] = self.tk.kanji.index(
                self.tk.question_entries[t].question)
            self.replies[t] = self.tk.meaning.index(
                self.tk.question_entries[t].reply)
            for i in range(N_POSSIBLE_REPLIES):
                self.possible_replies[t, i] = \
                    self.tk.meaning.index(
                        getattr(self.tk.question_entries[t],
                                f'possible_reply_{i}'))

            self.success[t] = self.questions[t] == self.replies[t]
