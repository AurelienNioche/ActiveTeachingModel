import numpy as np
from learner.act_r import ActR
np.seterr(all='raise')


class ActRDuoParam:

    def __init__(self, d, tau, s, r):

        # Decay parameter
        self.d = d
        # Retrieval threshold
        self.tau = tau
        # Noise in the activation levels
        self.s = s
        # Probability of choosing an option if random policy
        self.r = r


class ActRDuo(ActR):

    version = 0.0
    bounds = ('d', 0.001, 1.0), \
             ('tau', -1000, 1000), \
             ('s', 0.001, 100), \
             ('r', 0.0, 1.0)

    def __init__(self, tk, param=None, **kwargs):

        if param is not None:

            t_param = type(param)
            if t_param == dict:
                self.pr = ActRDuoParam(**param)

            elif t_param in (tuple, list, np.ndarray):
                self.pr = ActRDuoParam(*param)

            else:
                raise Exception(f"Type {type(param)} "
                                f"is not handled for parameters")

        super().__init__(tk=tk, **kwargs)

        self.items = np.arange(tk.n_item)
        self.p_random = self.pr.r

    def get_p_choices(self, data, use_p_correct=False):

        t_max = len(data.questions)

        p_choices = np.zeros(t_max)

        for t in range(t_max):

            question, reply = data.questions[t], data.replies[t]
            possible_rep = data.possible_replies[t]
            time = data.times[t]

            if use_p_correct:
                p = self._p_correct(question=question, reply=reply,
                                    possible_replies=possible_rep,
                                    time=time)

            else:
                p = self._p_choice(question=question, reply=reply,
                                   possible_replies=possible_rep,
                                   time=time)

            if p == 0:
                return None

            p_choices[t] = p
            self.learn(question=question, time=time)

        return p_choices
