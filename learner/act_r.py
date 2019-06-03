import numpy as np

from learner.generic import Learner


class ActRParam:

    def __init__(self, d, tau, s):

        # Decay parameter
        self.d = d
        # Retrieval threshold
        self.tau = tau
        # Noise in the activation levels
        self.s = s


class ActR(Learner):

    version = 2.1
    bounds = ('d', 0.000001, 1.0), ('tau', -5, 5), ('s', 0.0000001, 1)

    """
    A chunk is composed of:
    * a type (here: means)
    * several slots (here: slot 1: kanji, slot2: meaning)
    """

    def __init__(self, tk, param=None, verbose=False, track_p_recall=False):

        super().__init__()

        if param is None:
            pass  # ActR is used as abstract class
        elif type(param) == dict:
            self.pr = ActRParam(**param)
        elif type(param) in (tuple, list, np.ndarray):
            self.pr = ActRParam(*param)
        else:
            raise Exception(f"Type {type(param)} is not handled for parameters")

        self.tk = tk

        self.p_random = 1/self.tk.n_possible_replies

        # Time recording of presentations of chunks
        self.hist = np.ones(tk.t_max) * -99
        # self.time_presentation = [[] for _ in range(self.tk.n_item)]

        # Time counter
        self.t = 0

        # Do print
        self.verbose = verbose

        self.track_p_recall = track_p_recall

        if self.track_p_recall:
            self.p = np.zeros((self.tk.t_max, self.tk.n_item))

    def _activation_function(self, i):

        """The activation of a chunk is the sum of its base-level activation"""

        # noise = np.random.normal()
        b = self._base_level_learning_activation(i)  # + noise
        return b

    def _base_level_learning_activation(self, i):

        """The base-level activation measures how much time has elapsed since the jth use:"""

        # # noinspection PyTypeChecker
        # sum_a = np.sum([
        #     (self.t - t_presentation)**(-self.pr.d)
        #     for t_presentation in self.time_presentation[i]
        # ])
        #
        # if sum_a > 0:
        #     max_b = np.log(1 + np.sum([i ** self.pr.d for i in range(self.tk.t_max, 0, -1)]))
        #     b = np.log(1 + sum_a) / max_b
        # else:
        #     b = 0
        #
        # assert 0 <= b <= 1
        # return b
        time_presentation = np.asarray(self.hist == i).nonzero()[0]
        if not time_presentation.shape[0]:
            return -np.inf
        time_elapsed = self.t - time_presentation
        return np.log(np.power(time_elapsed, -self.pr.d).sum())

    def _sigmoid_function(self, a):

        """The probability of a chunk being above some retrieval threshold τ is"""

        x = (self.pr.tau - a) / (self.pr.s*np.square(2))

        # Avoid overflow
        if x < -10**2:  # 1 / (1+exp(-1000)) equals approx 1.
            return 1

        elif x > 700:  # 1 / (1+exp(700)) equals approx 0.
            return 0

        else:
            try:
                return 1 / (1 + np.exp(x))
            except FloatingPointError as e:
                print(f'x={x}, tau = {self.pr.tau}, a = {a}, s = {self.pr.s}')
                raise e

    def _update_time_presentation(self, question):

        # noinspection PyTypeChecker
        self.hist[self.t] = question
        # self.time_presentation[question].append(self.t)
        self.t += 1

    def p_recall(self, item):

        a = self._activation_function(item)
        p_retrieve = self._sigmoid_function(a)
        if self.verbose:
            print(f"t={self.t}, a_i: {a:.3f}, p_r: {p_retrieve:.3f}")
        return p_retrieve

    def _p_choice(self, question, reply, possible_replies=None):

        p_retrieve = self.p_recall(question)
        p_correct = self.p_random + p_retrieve*(1 - self.p_random)

        success = question == reply

        if success:
            return p_correct

        else:
            p_failure = (1-p_correct) / (self.tk.n_possible_replies - 1)
            return p_failure

    def _p_correct(self, question, reply, possible_replies=None):

        p_correct = self._p_choice(question=question, reply=question)

        correct = question == reply
        if correct:
            return p_correct

        else:
            return 1-p_correct

    def decide(self, question, possible_replies):

        p_r = self.p_recall(question)
        r = np.random.random()

        if p_r > r:
            reply = question
        else:
            reply = np.random.choice(possible_replies)

        if self.verbose:
            print(f't={self.t}: question {question}, reply {reply}')
        return reply

    def learn(self, question):

        self.questions.append(question)
        self._update_time_presentation(question)

        if self.track_p_recall:
            for i in range(self.tk.n_item):
                self.p[self.t - 1, i] = self.p_recall(i)

    def unlearn(self):

        # try:
        #     last_question = self.questions.pop()
        # except IndexError:
        #     raise AssertionError("I can not unlearn something that has not been learned!")

        self.t -= 1

        self.hist[self.t] = -99


# class ActROriginal(ActR):
# #
# #     def __init__(self, tk, param=None, verbose=False):
# #
# #         super().__init__(tk=tk, param=param, verbose=verbose)
# #
# #     def _base_level_learning_activation(self, i):
# #
# #         """The base-level activation measures how much time has elapsed since the jth use:"""
# #
# #         # # noinspection PyTypeChecker
# #         # sum_a = np.sum([
# #         #     (self.t - t_presentation)**(-self.pr.d)
# #         #     for t_presentation in self.time_presentation[i]
# #         # ])
# #         #
# #         # b = np.log(sum_a) if sum_a > 0 else -np.inf
# #         # return b
# #         sum_a = np.sum((self.t - np.asarray(self.a == i).nonzero()[0]) ** (-self.pr.d))
# #         b = np.log(sum_a) if sum_a > 0 else -np.inf
# #         return b
