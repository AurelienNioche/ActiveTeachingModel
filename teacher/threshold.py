import numpy as np
from teacher.psychologist import Psychologist, Learner


class Threshold:

    def __init__(self, n_item, learnt_threshold,
                 bounds, grid_size,
                 is_item_specific,
                 param=None):

        self.psychologist = Psychologist(
            n_item=n_item,
            bounds=bounds,
            grid_size=grid_size,
            is_item_specific=is_item_specific)
        self.n_item = n_item
        self.learnt_threshold = learnt_threshold
        self.param = param

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

    def _select_item(self, now):

        p, seen = self.psychologist.p_seen(now, param=self.param)

        min_p = np.min(p)

        if np.sum(seen) == self.n_item or min_p <= self.learnt_threshold:
            item_idx = np.arange(self.n_item)[seen][np.argmin(p)]

        else:
            item_idx = np.argmin(seen)

        return item_idx

    @classmethod
    def create(cls, tk, omniscient):

        return cls(
            n_item=tk.n_item,
            learnt_threshold=tk.learnt_threshold,
            bounds=tk.bounds,
            grid_size=tk.grid_size,
            is_item_specific=tk.is_item_specific,
            param=tk.param if omniscient else None)
