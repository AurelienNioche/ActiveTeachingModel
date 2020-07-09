from abc import abstractmethod


class Learner:

    @abstractmethod
    def p_seen(self, param, is_item_specific, now):
        raise NotImplementedError

    @abstractmethod
    def p(self, item, param, now, is_item_specific):
        raise NotImplementedError

    @abstractmethod
    def update(self, item, timestamp):
        raise NotImplementedError

    @abstractmethod
    def log_lik_grid(self, item, grid_param, response, timestamp):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def log_lik(param, hist, success, timestamp):
        raise NotImplementedError

    @classmethod
    def inv_log_lik(cls, param, hist, success, timestamp):
        return - cls.log_lik(param=param, hist=hist, success=success,
                             timestamp=timestamp)
