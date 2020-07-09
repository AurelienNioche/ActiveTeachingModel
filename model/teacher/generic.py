from abc import abstractmethod


class Teacher:

    @abstractmethod
    def ask(self, now, last_was_success=None, last_time_reply=None,
            idx_last_q=None):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create(cls, tk, omniscient):
        raise NotImplementedError
