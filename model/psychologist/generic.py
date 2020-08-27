from abc import abstractmethod


class Psychologist:

    @abstractmethod
    def update(self, item, response, timestamp):
        raise NotImplementedError

    @abstractmethod
    def p_seen(self, now, param=None):
        raise NotImplementedError

    @abstractmethod
    def inferred_learner_param(self):
        raise NotImplementedError

    @abstractmethod
    def p(self, param, item, now):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs):
        raise NotImplementedError
