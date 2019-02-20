import numpy as np

from task import task


class TeacherMemory:

    def __init__(self):

        self.questions = np.zeros(task.t_max, dtype=int)
        self.replies = np.zeros(task.t_max, dtype=int)
        self.successes = np.zeros(task.t_max, dtype=bool)


class Teacher:

    def __init__(self):

        self.question = None

        self.memory = TeacherMemory()

    def choose_question(self, t):

        # if t < exercise.t_max:
        #     self.question = np.random.randint(int(exercise.n/2))
        #
        # else:
        #     self.question = np.random.randint(int(exercise.n/2), exercise.n)
        self.question = np.random.randint(task.n)

        self.memory.questions[t] = self.question

        return self.question

    def evaluate(self, t, reply):

        correct_answer = task.items[self.question]
        success = correct_answer == reply

        self.memory.replies[t] = reply
        self.memory.successes[t] = success

        return correct_answer, success

    def summarize(self):

        return self.memory.successes
