import numpy as np

from task.models import User, Question
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

    items = [q.question for q in question_entries]

    kanjis = list(np.unique(items))
    meanings = []

    for k in kanjis:
        meanings.append(Question.objects.filter(user_id=user_id, question=k)[0].correct_answer)

    if verbose:
        print(f"Kanji used: {kanjis}")
        print(f"Corresponding meanings: {meanings}")

    return question_entries, kanjis, meanings


def get(user_id):

    question_entries, kanjis, meanings = task_features(user_id=user_id)

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
