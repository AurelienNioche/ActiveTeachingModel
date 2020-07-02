from abc import abstractmethod


class Learner:

    @abstractmethod
    def p_seen(self, param, is_item_specific, now):
        raise NotImplementedError

    @abstractmethod
    def log_lik(self, item, grid_param, response, timestamp):
        raise NotImplementedError

    @abstractmethod
    def p(self, item, param, now, is_item_specific):
        raise NotImplementedError

    @abstractmethod
    def update(self, item, timestamp):
        raise NotImplementedError
