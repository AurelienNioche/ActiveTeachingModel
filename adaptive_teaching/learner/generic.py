import numpy as np

# np.seterr(all='raise')


class Learner:

    version = 0.0
    bounds = {
        '<name of parameter>': (0.0, 1.0),
    }

    def __init__(self):
        pass

    @classmethod
    def generate_random_parameters(cls):
        return {t[0]: np.random.uniform(t[1], t[2]) for t in cls.bounds}

    def update(self, item):
        raise NotImplementedError

    def cancel_update(self):
        raise NotImplementedError

    def p_recall(self, item):
        raise NotImplementedError
