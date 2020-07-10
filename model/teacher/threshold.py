import numpy as np
from .generic import Teacher


class Threshold(Teacher):

    def __init__(self, n_item, learnt_threshold, psychologist):

        self.psychologist = psychologist
        self.n_item = n_item
        self.learnt_threshold = learnt_threshold

        self.new_item_counter = 0

        self.limit_new_item = 5

    def _select_item(self, now):

        p, seen = self.psychologist.p_seen(now)

        min_p = np.min(p)

        if np.sum(seen) == self.n_item or min_p <= self.learnt_threshold \
                or self.new_item_counter >= self.limit_new_item:
            item_idx = np.arange(self.n_item)[seen][np.argmin(p)]
            self.new_item_counter = 0
        else:
            item_idx = np.argmin(seen)
            self.new_item_counter += 1

        return item_idx

    def ask(self, now, last_was_success=None, last_time_reply=None,
            idx_last_q=None):
        if idx_last_q is None:
            item_idx = 0

        else:

            self.psychologist.update(item=idx_last_q,
                                     response=last_was_success,
                                     timestamp=last_time_reply)

            item_idx = self._select_item(now)

        return item_idx

    @classmethod
    def create(cls, tk, omniscient):

        psy = tk.psychologist_model.create(tk=tk, omniscient=omniscient)

        return cls(
            n_item=tk.n_item,
            learnt_threshold=tk.learnt_threshold,
            psychologist=psy)
