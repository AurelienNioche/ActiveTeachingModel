from abc import abstractmethod


class Teacher:

    @abstractmethod
    def ask(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs):
        raise NotImplementedError
