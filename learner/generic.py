import numpy as np

# np.seterr(all='raise')


class Learner:

    version = 0.0
    bounds = ('<name of parameter>', 0.0000001, 1.0),

    def __init__(self):
        self.questions = []

    def decide(self, question, possible_replies, time=None):
        raise NotImplementedError

    def learn(self, question, time=None):
        raise NotImplementedError

    def unlearn(self):
        raise NotImplementedError

    def _p_choice(self, question, reply, possible_replies, time=None):
        raise NotImplementedError

    def _p_correct(self, question, reply, possible_replies, time=None):
        raise NotImplementedError

    def p_recall(self, item, time=None):
        raise NotImplementedError

    def get_p_choices(self, data, fit_param=None):

        t_max = len(data.questions)

        p_choices = np.zeros(t_max)

        for t in range(t_max):

            question, reply = data.questions[t], data.replies[t]
            possible_rep = data.possible_replies[t]

            if fit_param.get('use_p_correct'):
                p = self._p_correct(question=question, reply=reply,
                                    possible_replies=possible_rep)

            else:
                p = self._p_choice(question=question, reply=reply,
                                   possible_replies=possible_rep)

            if p == 0:
                return None

            p_choices[t] = p
            self.learn(question=question)

        return p_choices
