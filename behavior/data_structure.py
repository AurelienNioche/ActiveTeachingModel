import numpy as np


class Data:

    def __init__(self, t_max, questions=None,
                 replies=None, possible_replies=None,
                 times=None, n_possible_replies=None):

        self.questions = np.zeros(t_max, dtype=int)
        self.replies = np.zeros(t_max, dtype=int)
        self.times = np.zeros(t_max, dtype=int)  # Second

        if n_possible_replies is not None:
            self.possible_replies = np.zeros(
                (t_max, n_possible_replies), dtype=int)
            if possible_replies is not None:
                self.possible_replies[:] = possible_replies

        else:
            self.possible_replies = [None for _ in range(t_max)]

        if questions is not None:
            self.questions[:] = questions

        if replies is not None:
            self.replies[:] = replies

        if times is not None:
            self.times[:] = times


class Task:

    def __init__(self, t_max, n_possible_replies=None):

        self.t_max = t_max
        self.n_possible_replies = n_possible_replies
