import numpy as np


class TeacherMemory:

    def __init__(self, task):

        self.questions = np.zeros(task.n_iteration, dtype=int) * -1
        self.replies = np.zeros(task.n_iteration, dtype=int) * -1
        self.successes = np.zeros(task.n_iteration, dtype=bool) * -1


class Teacher:

    def __init__(self, task, plan_questions_in_advance=False):

        self.task = task
        self.memory = TeacherMemory(task)

        if plan_questions_in_advance:
            self._plan_questions_in_advance()

        self.questions_already_planned = plan_questions_in_advance

    def _plan_questions_in_advance(self):

        self.memory.questions[:] = np.random.choice(self.task.questions, size=self.task.n_iteration)

    def _choose_question(self, t):

        # if t < exercise.n_iteration:
        #     self.question = np.random.randint(int(exercise.n/2))
        #
        # else:
        #     self.question = np.random.randint(int(exercise.n/2), exercise.n)

        return np.random.choice(self.task.questions)

    def ask_question(self, t):

        if not self.questions_already_planned:

            question = self._choose_question(t)
            self.memory.questions[t] = question

        else:
            question = self.memory.questions[t]

        return question

    def evaluate(self, t, reply):

        success = reply == self.memory.questions[t]

        self.memory.replies[t] = reply
        self.memory.successes[t] = success

    def summarize(self):

        return self.memory.successes
