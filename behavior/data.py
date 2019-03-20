import numpy as np

from task.models import User, Question, Kanji
from task.parameters import n_possible_replies


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

            to_print = f't:{t}, question: {q.question}, reply: {q.reply}, correct answer: {q.correct_answer}, ' \
                f'success: {"Yes" if success else "No"}, reaction time: {reaction_time} ms'
            print(to_print)

        print()


def task_features(user_id, verbose=False):

    question_entries = [q for q in Question.objects.filter(user_id=user_id).order_by('t')]

    answers = []

    for q in question_entries:

        for i in range(n_possible_replies):
            answer = getattr(q, f'possible_reply_{i}')
            answers.append(answer)

    u_answers = np.unique(answers)

    meanings = list(u_answers)
    kanjis = []
    for m in meanings:
        try:
            kanji = Question.objects.filter(correct_answer=m)[0].question

        except IndexError:
            kanji = Kanji.objects.filter(meaning=m)[0].kanji
        kanjis.append(kanji)

    if verbose:
        print(f"Kanji used: {kanjis}")
        print(f"Corresponding meanings: {meanings}")

    return question_entries, kanjis, meanings


def get(user_id, verbose=False):

    question_entries, kanjis, meanings = task_features(user_id=user_id, verbose=verbose)

    n_items = len(kanjis)

    t_max = question_entries[-1].t

    questions = np.zeros(t_max, dtype=int)
    replies = np.zeros(t_max, dtype=int)
    possible_replies = np.zeros((t_max, n_possible_replies), dtype=int)
    success = np.zeros(t_max, dtype=int)

    for t in range(t_max):

        questions[t] = kanjis.index(question_entries[t].question)
        replies[t] = meanings.index(question_entries[t].reply)
        for i in range(n_possible_replies):
            possible_replies[t, i] = meanings.index(getattr(question_entries[t], f'possible_reply_{i}'))

        success[t] = questions[t] == replies[t]

    return questions, replies, n_items, possible_replies, success
