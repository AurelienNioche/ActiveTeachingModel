import numpy as np
from abc import abstractmethod


class Psychologist:

    @abstractmethod
    def update(self, item, response, timestamp):
        raise NotImplementedError

    @abstractmethod
    def update_learner(self, item, timestamp):
        raise NotImplementedError

    @abstractmethod
    def p_seen(self, now):
        raise NotImplementedError

    @abstractmethod
    def inferred_learner_param(self):
        raise NotImplementedError

    @abstractmethod
    def p(self, param, item, now):
        raise NotImplementedError

    @classmethod
    def generate_param(cls, param, bounds, n_item):

        if isinstance(param, str):
            if param in ("heterogeneous", "het"):
                param = np.zeros((n_item, len(bounds)))
                for i, b in enumerate(bounds):
                    param[:, i] = np.random.uniform(b[0], b[1], size=n_item)

            else:
                raise ValueError
        else:
            param = np.asarray(param)
        return param

    @classmethod
    @abstractmethod
    def create(cls, tk, omniscient):
        raise NotImplementedError
