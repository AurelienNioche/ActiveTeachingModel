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

    def update(self, item, response):
        raise NotImplementedError

    def cancel_update(self):
        raise NotImplementedError

    def p(self, item):
        raise NotImplementedError

    def p_grid(self, grid_param, i):
        raise NotImplementedError

    def forgetting_rate_and_p_seen(self):
        raise NotImplementedError

    def forgetting_rate_and_p_all(self):
        raise NotImplementedError

    def set_param(self, param):
        raise NotImplementedError
