import numpy as np


class TeacherMemory:

    def __init__(self, task):

        self.questions = np.zeros(task.t_max, dtype=int) * -1
        self.replies = np.zeros(task.t_max, dtype=int) * -1
        self.successes = np.zeros(task.t_max, dtype=bool) * -1


class Teacher:

    def __init__(self, task, plan_questions_in_advance=False):

        self.task = task
        self.memory = TeacherMemory(task)

        if plan_questions_in_advance:
            self.plan_questions_in_advance()

        self.questions_already_planned = plan_questions_in_advance

    def plan_questions_in_advance(self):

        self.memory.questions[:] = np.random.choice(self.task.questions, size=self.task.t_max)

    def ask_question(self, t):

        if not self.questions_already_planned:

            question = self.choose_question(t)
            self.memory.questions[t] = question

        else:
            question = self.memory.questions[t]

        return question

    def choose_question(self, t):

        # if t < exercise.t_max:
        #     self.question = np.random.randint(int(exercise.n/2))
        #
        # else:
        #     self.question = np.random.randint(int(exercise.n/2), exercise.n)

        return np.random.choice(self.task.questions)

    def evaluate(self, t, reply):

        correct_answer = self.task.get_reply(self.memory.questions[t])
        success = correct_answer == reply

        self.memory.replies[t] = reply
        self.memory.successes[t] = success

        return correct_answer, success

    def summarize(self):

        return self.memory.successes
