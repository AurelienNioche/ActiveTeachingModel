import numpy as np

from task import exercise


class TeacherMemory:

    def __init__(self):

        self.questions = np.zeros(exercise.t_max, dtype=int)
        self.replies = np.zeros(exercise.t_max, dtype=int)
        self.successes = np.zeros(exercise.t_max, dtype=bool)


class Teacher:

    def __init__(self):

        self.question = None

        self.memory = TeacherMemory()

    def choose_question(self, t):

        self.question = np.random.randint(exercise.n)

        self.memory.questions[t] = self.question

        return self.question

    def evaluate(self, t, reply):

        correct_answer = exercise.items[self.question]
        success = correct_answer == reply

        self.memory.replies[t] = reply
        self.memory.successes[t] = success

        return correct_answer, success

    def summarize(self):

        return self.memory.successes
