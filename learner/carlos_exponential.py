import math
import numpy as np
import sys

from learner.generic import Learner


class ExponentialParam:
    def __init__(self, alpha, beta, n_0):
        self.alpha = alpha
        self.beta = beta
        self.n_0 = n_0


class Exponential(Learner):

    def __init__(self, param, tk, verbose=True):
        self.verbose = verbose
        self.tk = tk

        if param is None:
            pass
        elif type(param) == dict:
            self.pr = ExponentialParam(**param)
        elif type(param) in (tuple, list, np.ndarray):
            self.pr = ExponentialParam(*param)
        else:
            raise Exception(f"Type {type(param)} is not handled for parameters")

        self.hist = np.zeros(self.tk.t_max, dtype=int)
        self.success = np.zeros(self.tk.t_max, dtype=bool)
        self.t = 0
        self.last_reply = None
        self.last_forgetting_rate = self.pr.n_0

        super().__init__()

    def decide(self, question, possible_replies):

        p_r = self.p_recall(question)
        r = np.random.random()

        if p_r > r:
            reply = question
        else:
            reply = np.random.choice(possible_replies)

        if self.verbose:
            print(f't={self.t}: question {question}, reply {reply}')

        self.last_reply = reply

        return reply

    def p_recall(self, question):
        # m(t) = exp(-n(t)(t-t_{last review})

        last_score = None

        occurrences = (self.hist == question).nonzero()[0]  # returns indexes
        if len(occurrences):
            t_last_review = self.t - occurrences[-1]
        else:
            t_last_review = 0

        # p_r = 0.5
        # r = np.random.random()

        # last recall of the item gud??

        success = self.success[self.hist == question]
        if len(success):
            if self.success[self.hist == question][-1]:
                forgetting_rate = (1 - self.pr.alpha) \
                                  * self.last_forgetting_rate
            else:
                forgetting_rate = (1 + self.pr.beta) \
                                  * self.last_forgetting_rate
        else:
            forgetting_rate = self.pr.n_0

        self.last_forgetting_rate = forgetting_rate

        p_r = math.exp(-forgetting_rate * (self.t - t_last_review))
        # func = n(t) forgetting rate, which depends on the past reviews

        return p_r

    def learn(self, question):
        self.hist[self.t] = question
        self.success[self.t] = self.last_reply == question
        self.t += 1


# the question now has been asked, was that question recalled or not?
