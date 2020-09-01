from abc import abstractmethod


class Teacher:

    @abstractmethod
    def ask(self, *args, **kwargs):
        raise NotImplementedError
