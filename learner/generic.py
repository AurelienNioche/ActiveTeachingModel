import numpy as np

# np.seterr(all='raise')


class Learner:

    version = 0.0
    bounds = ('<name of parameter>', 0.0000001, 1.0),

    def __init__(self):
        pass

    def decide(self, question, possible_replies, time=None):
        """Expected return from specific learner: reply"""
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
        """Expected return from specific learner: p_r"""
        raise NotImplementedError

    def get_p_choices(self, data, use_p_correct=False, stop_if_zero=True):

        t_max = len(data.questions)

        p_choices = np.zeros(t_max)

        for t in range(t_max):

            question, reply = data.questions[t], data.replies[t]
            time = data.times[t]

            if hasattr(data, 'possible_replies'):
                possible_rep = data.possible_replies[t]
            else:
                possible_rep = None
                use_p_correct = True

            if hasattr(data, 'first_presentation'):
                first_pr = data.first_presentation[t]
                if first_pr:
                    self.learn(question=question, time=time)

            if use_p_correct:
                p = self._p_correct(question=question, reply=reply,
                                    possible_replies=possible_rep,
                                    time=time)

            else:
                p = self._p_choice(question=question, reply=reply,
                                   possible_replies=possible_rep,
                                   time=time)

            if p == 0 and stop_if_zero:
                return None

            p_choices[t] = p
            self.learn(question=question, time=time)

        return p_choices

    @classmethod
    def generate_random_parameters(cls):

        return {t[0]: np.random.uniform(t[1], t[2]) for t in cls.bounds}

    def set_parameters(self, param):

        if param is None:
            pass

        elif type(param) == dict:
            for k, v in param.items():
                setattr(self, k, v)

        elif type(param) in (tuple, list, np.ndarray):

            for tup, v in zip(self.bounds, param):
                setattr(self, tup[0], v)
        else:
            raise Exception(f"Type {type(param)} "
                            f"is not handled for parameters")
