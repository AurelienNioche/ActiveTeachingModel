import numpy as np
# from scipy.stats import norm

debug = False
# np.seterr(all='raise')


class Task:

    def __init__(self, t_max, n_possible_replies, n_items, c_graphic=None, c_semantic=None):

        self.t_max = t_max
        self.n_possible_replies = n_possible_replies
        self.n_items = n_items
        self.c_graphic = c_graphic
        self.c_semantic = c_semantic


class Exp:

    def __init__(self, questions, replies, possible_replies):

        self.questions = questions
        self.replies = replies
        self.possible_replies = possible_replies


class Learner:

    def __init__(self):
        self.questions = []

    def decide(self, question, possible_replies):
        return 0

    def learn(self, question):
        pass

    def unlearn(self):
        pass

    @property
    def name(self):
        return type(self).__name__

    def _p_choice(self, question, reply, possible_replies):
        return -1

    def _p_correct(self, question, reply, possible_replies):
        return -1

    def p_recall(self, item):

        return 0.5

    def get_p_choices(self, exp, fit_param=None):

        xp = Exp(**exp)

        t_max = len(xp.questions)

        p_choices = np.zeros(t_max)

        for t in range(t_max):

            question, reply = xp.questions[t], xp.replies[t]
            possible_rep = xp.possible_replies[t]

            if fit_param is not None and fit_param['use_p_correct'] is True:
                p = self._p_correct(question=question, reply=reply, possible_replies=possible_rep)

            else:
                p = self._p_choice(question=question, reply=reply, possible_replies=possible_rep)

            if p == 0:
                return None

            p_choices[t] = p
            self.learn(question=question, reply=reply)

        return p_choices
