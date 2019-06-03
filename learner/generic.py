import numpy as np

# np.seterr(all='raise')


class Learner:

    def __init__(self):
        self.questions = []

    def decide(self, question, possible_replies):
        return 0

    def learn(self, question):
        pass

    def unlearn(self):
        pass

    def _p_choice(self, question, reply, possible_replies):
        return -1

    def _p_correct(self, question, reply, possible_replies):
        return -1

    def p_recall(self, item):
        return 0.5

    def get_p_choices(self, data, fit_param=None):

        t_max = len(data.questions)

        p_choices = np.zeros(t_max)

        for t in range(t_max):

            question, reply = data.questions[t], data.replies[t]
            possible_rep = data.possible_replies[t]

            if fit_param.get('use_p_correct'):
                p = self._p_correct(question=question, reply=reply, possible_replies=possible_rep)

            else:
                p = self._p_choice(question=question, reply=reply, possible_replies=possible_rep)

            if p == 0:
                return None

            p_choices[t] = p
            self.learn(question=question)

        return p_choices
