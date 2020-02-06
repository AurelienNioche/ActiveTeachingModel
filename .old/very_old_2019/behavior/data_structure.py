import numpy as np


class Data:

    def __init__(self, n_iteration, questions=None,
                 replies=None, possible_replies=None,
                 times=None, n_possible_replies=None):

        self.questions = np.zeros(n_iteration, dtype=int)
        self.replies = np.zeros(n_iteration, dtype=int)
        self.times = np.zeros(n_iteration, dtype=int)  # Second

        if n_possible_replies is not None:
            self.possible_replies = np.zeros(
                (n_iteration, n_possible_replies), dtype=int)
            if possible_replies is not None:
                self.possible_replies[:] = possible_replies

        else:
            self.possible_replies = [None for _ in range(n_iteration)]

        if questions is not None:
            self.questions[:] = questions

        if replies is not None:
            self.replies[:] = replies

        if times is not None:
            self.times[:] = times


class Task:

    def __init__(self, n_iteration, n_item=None, n_possible_replies=None):

        self.n_iteration = n_iteration
        self.n_item = n_item
        self.n_possible_replies = n_possible_replies
