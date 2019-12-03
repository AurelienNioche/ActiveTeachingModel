import math

import numpy as np

from learner.generic import Learner


class DecayParam:
    def __init__(self, difficulty):
        # self.difficulty = difficulty
        pass


class Decay(Learner):

    def __init__(self, param, tk, verbose=True):
        self.verbose = verbose
        self.tk = tk

        if param is None:
            pass
        elif type(param) == dict:
            self.pr = DecayParam(**param)
        elif type(param) in (tuple, list, np.ndarray):
            self.pr = DecayParam(*param)
        else:
            raise Exception(
                f"Type {type(param)} is not handled for parameters")

        self.hist = np.zeros(self.tk.n_iteration, dtype=int)
        self.success = np.zeros(self.tk.n_iteration, dtype=bool)
        self.t = 0
        self.last_p_r = 1
        self.last_reply = None

        super().__init__()

    def decide(self, item, possible_replies, time=None):

        p_r = self.p_recall(item)
        r = np.random.random()

        if p_r > r:
            reply = item
        else:
            reply = np.random.choice(possible_replies)

        if self.verbose:
            print(f't={self.t}: question {item}, reply {reply}')

        self.last_reply = reply

        return reply

    def p_recall(self, question, time=None):
        """
        Fun stuffz
        """

        p_r = math.exp(- (1 / (0.01 * self.last_p_r + 0.00000001)) * self.t)
        - math.exp(- success + history)
        self.last_p_r = p_r

        return p_r

    def learn(self, item):
        self.hist[self.t] = item
        self.success[self.t] = self.last_reply == item
        self.t += 1
