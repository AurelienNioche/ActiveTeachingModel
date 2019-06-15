from task.parameters import N_POSSIBLE_REPLIES

import numpy as np


class Data:

    def __init__(self, t_max):

        self.questions = np.zeros(t_max, dtype=int)
        self.replies = np.zeros(t_max, dtype=int)
        self.possible_replies = np.zeros(
            (t_max, N_POSSIBLE_REPLIES), dtype=int)


class Task:

    def __init__(self, t_max):

        self.t_max = t_max
        self.n_possible_replies = N_POSSIBLE_REPLIES
