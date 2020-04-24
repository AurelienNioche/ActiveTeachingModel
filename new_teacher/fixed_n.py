import numpy as np

from . abstract import Teacher


class FixedNTeacher(Teacher):

    def __init__(self, n_item, n_iter_per_ss, n_iter_between_ss,
                 optimal_n):

        super().__init__(
            n_item=n_item,
            n_iter_per_ss=n_iter_per_ss,
            n_iter_between_ss=n_iter_between_ss)

        self.optimal_n = optimal_n

        self.items = np.arange(self.optimal_n)

        self.c = 0

    def ask(self):

        item = self.items[self.c]
        self.c = (self.c + 1) % self.optimal_n
        return item





