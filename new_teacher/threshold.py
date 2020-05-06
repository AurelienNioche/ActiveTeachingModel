import numpy as np

from . abstract import Teacher
from . psychologist import Psychologist


class ThresholdTeacher(Teacher):

    def __init__(self, n_item, n_iter_per_ss, n_iter_between_ss,
                 param, learnt_threshold):

        super().__init__(
            n_item=n_item,
            n_iter_per_ss=n_iter_per_ss,
            n_iter_between_ss=n_iter_between_ss)

        self.n_pres = np.zeros(n_item, dtype=int)
        self.delta = np.zeros(n_item, dtype=int)

        self.c_iter_session = 0

        self.items = np.arange(n_item)

        self.param = param
        self.learnt_threshold = learnt_threshold

    def ask(self):

        seen = self.n_pres[:] > 0
        n_seen = np.sum(seen)

        if n_seen == 0:
            items_selected = self.items

        else:
            fr = self.param[0] * (1 - self.param[1]) ** (self.n_pres[seen] - 1)
            p = np.exp(-fr * self.delta[seen])

            min_p = np.min(p)

            if n_seen == self.n_item or min_p <= self.learnt_threshold:
                items_selected = self.items[seen][p[:] == min_p]

            else:
                unseen = np.logical_not(seen)
                items_selected = self.items[unseen]

        item = np.random.choice(items_selected)
        self.update(item)
        return item

    def update(self, item):

        self.n_pres[item] += 1

        # Increment delta for all items
        self.delta[:] += 1
        # ...except the one for the selected design that equal one
        self.delta[item] = 1

        self.c_iter_session += 1
        if self.c_iter_session >= self.n_iter_per_ss:
            self.delta[:] += self.n_iter_between_ss
            self.c_iter_session = 0


class ThresholdPsychologist(ThresholdTeacher):

    def __init__(self, n_item, n_iter_per_ss, n_iter_between_ss,
                 learnt_threshold, n_ss, param):
        super().__init__(n_item=n_item,
                         n_iter_per_ss=n_iter_per_ss,
                         param=param,
                         n_iter_between_ss=n_iter_between_ss,
                         learnt_threshold=learnt_threshold)
        self.psychologist = Psychologist(
            n_item=n_item,
            n_iter_per_ss=n_iter_per_ss,
            n_iter_between_ss=n_iter_between_ss,
            n_ss=n_ss,
            true_param=param)

    def ask(self):
        self.param = self.psychologist.pm
        return super().ask()

    def update(self, item):

        super().update(item)
        self.psychologist.update(item=item, response=None)
