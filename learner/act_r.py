import numpy as np

from learner.generic import Learner


class ActRParam:
    """
    :param d: decay rate of the forgetting curve. Range 0.001 to 1.0
    :param tau: retrieval threshold. Range not determined
    :param s: noise in the activation levels. Range s > 0

    """

    def __init__(self, d, tau, s):

        self.d = d
        self.tau = tau
        self.s = s


class ActR(Learner):

    version = 2.2
    bounds = ('d', 0.001, 1.0), \
             ('tau', -1, 1), \
             ('s', 0.001, 1)

    """
    A chunk is composed of:
        * a type (here: means)
        * several slots (here: slot 1: kanji, slot2: meaning)
    """

    def __init__(self, tk, param=None, verbose=False):

        super().__init__()

        if param is None:
            pass  # ActR is used as abstract class
        elif type(param) == dict:
            self.pr = ActRParam(**param)
        elif type(param) in (tuple, list, np.ndarray):
            self.pr = ActRParam(*param)
        else:
            raise Exception(f"Type {type(param)} "
                            f"is not handled for parameters")

        self.tk = tk

        if self.tk.n_possible_replies is not None:
            self.p_random = 1/self.tk.n_possible_replies
        else:
            self.p_random = 0

        # History of presentation
        self.hist = np.ones(tk.t_max) * -99

        # Time counter
        self.t = 0

        # Options
        self.verbose = verbose

        # For continuous time
        self.times = np.zeros(self.tk.t_max)

        self._bkp_presentation_effect = {}

    def _activation_function(self, i, time=None,
                             time_index=None):

        """The activation of a chunk is the sum of its base-level activation"""

        # noise = np.random.normal()
        b = self._base_level_learning_activation(i,
                                                 time=time,
                                                 time_index=time_index)
        # + noise
        return b

    def _base_level_learning_activation(self, i, time=None,
                                        time_index=None):
        """
        The base-level activation measures
        how much time has elapsed since the j^th use:
        """

        pe = self._presentation_effect(i,
                                       time=time,
                                       time_index=time_index)
        if pe > 0.0001:
            return np.log(pe)
        else:
            return - np.inf

    def _presentation_effect(self, i, time=None,
                             time_index=None):

        try:
            if time is not None:
                return self._bkp_presentation_effect[time][i]
            elif time_index is not None:
                return self._bkp_presentation_effect[time_index][i]
            else:
                return self._bkp_presentation_effect[self.t][i]
        except KeyError:
            pass

        i_presented = self.hist == i

        if time is not None:

            try:

                in_past = self.times <= time
            except FloatingPointError as e:
                print('time', time)
                print('times', self.times)
                raise e

            time_presentation = self.times[i_presented * in_past]
            if not time_presentation.shape[0]:
                return 0  # This item has never been seen
            time_elapsed = np.asarray(time - time_presentation, dtype=float)
            time_elapsed[time_elapsed == 0] = 0.0001  # To avoid div by 0

        elif time_index is not None:

            time_presentation = \
                i_presented[:time_index+1].nonzero()[0]

            if not time_presentation.shape[0]:
                return 0  # This item has never been seen
            time_elapsed = \
                np.asarray(time_index - time_presentation, dtype=float)
            time_elapsed[time_elapsed == 0] = 0.0001  # To avoid div by 0

        else:
            time_presentation = i_presented[:].nonzero()[0]
            if not time_presentation.shape[0]:
                return 0  # This item has never been seen
            time_elapsed = self.t - time_presentation

        # Presentation effect
        pe = np.power(time_elapsed, -self.pr.d).sum()

        # # Save it!
        if time is not None:
            t = time

        elif time_index is not None:
            t = time_index

        else:
            t = self.t

        if time not in self._bkp_presentation_effect.keys():
            self._bkp_presentation_effect[t] = {}
        self._bkp_presentation_effect[t][i] = pe

        return pe

    def _sigmoid_function(self, a):

        """The probability of a chunk being above
        some retrieval threshold τ is
        """

        x = (self.pr.tau - a) / (self.pr.s*np.square(2))

        # Avoid overflow
        if x < -10**2:  # 1 / (1+exp(-1000)) equals approx 1.
            return 1

        elif x > 700:  # 1 / (1+exp(700)) equals approx 0.
            return 0

        try:
            return 1 / (1 + np.exp(x))
        except FloatingPointError as e:
            print(f'x={x}, tau = {self.pr.tau}, a = {a}, s = {self.pr.s}')
            raise e

    def p_recall(self, item, time=None, time_index=None):

        a = self._activation_function(item,
                                      time=time,
                                      time_index=time_index)
        p_retrieve = self._sigmoid_function(a)
        if self.verbose:
            print(f"t={self.t}, a_i: {a:.3f}, p_r: {p_retrieve:.3f}")
        return p_retrieve

    def _p_choice(self, question, reply, possible_replies=None,
                  time=None, time_index=None):

        success = question == reply

        p_recall = self.p_recall(question,
                                 time=time,
                                 time_index=time_index)

        # If number of possible replies is defined
        if self.tk.n_possible_replies is not None:
            p_correct = self.p_random + p_recall*(1 - self.p_random)

            if success:
                p_choice = p_correct

            else:
                p_choice = (1-p_correct) / (self.tk.n_possible_replies - 1)

        else:
            # Ignore in computation of reply the alternatives
            # p_choice = p_recall if success else 1-p_recall
            p_correct = self.p_random + p_recall * (1 - self.p_random)

            if success:
                p_choice = p_correct

            else:
                p_choice = (1 - p_correct)

        return p_choice

    def _p_correct(self, question, reply, possible_replies=None,
                   time=None, time_index=None):

        p_correct = self._p_choice(question=question, reply=question,
                                   time=time, time_index=time_index)

        correct = question == reply
        if correct:
            return p_correct

        else:
            return 1-p_correct

    def decide(self, question, possible_replies, time=None,
               time_index=None):

        p_r = self.p_recall(question,
                            time=time,
                            time_index=time_index)
        r = np.random.random()

        if p_r > r:
            reply = question
        else:
            reply = np.random.choice(possible_replies)

        if self.verbose:
            print(f't={self.t}: question {question}, reply {reply}')
        return reply

    def learn(self, question, time=None, time_index=None):

        if time_index is not None:
            self.hist[time_index] = question
            self.times[time_index] = time
        else:
            self.hist[self.t] = question
            self.times[self.t] = time
            self.t += 1

    def unlearn(self, time_index=None):

        # try:
        #     last_question = self.questions.pop()
        # except IndexError:
        #     raise AssertionError("I can not unlearn something
        #     that has not been learned!")

        if time_index is not None:
            self.hist[time_index] = -99
            self.times[time_index] = -1

            try:
                del self._bkp_presentation_effect[time_index]
            except KeyError:
                pass

        else:
            self.t -= 1

            self.hist[self.t] = -99
            self.times[self.t] = -1

            try:
                del self._bkp_presentation_effect[self.t]
            except KeyError:
                pass


# class ActROriginal(ActR):
# #
# #     def __init__(self, tk, param=None, verbose=False):
# #
# #         super().__init__(tk=tk, param=param, verbose=verbose)
# #
# #     def _base_level_learning_activation(self, i):
# #
# #         """The base-level activation measures how much time has elapsed
# #         since the jth use:"""
# #
# #         # # noinspection PyTypeChecker
# #         # sum_a = np.sum([
# #         #     (self.t - t_presentation)**(-self.pr.d)
# #         #     for t_presentation in self.time_presentation[i]
# #         # ])
# #         #
# #         # b = np.log(sum_a) if sum_a > 0 else -np.inf
# #         # return b
# #         sum_a = np.sum((self.t - np.asarray(self.a == i).nonzero()[0])
#           ** (-self.pr.d))
# #         b = np.log(sum_a) if sum_a > 0 else -np.inf
# #         return b
