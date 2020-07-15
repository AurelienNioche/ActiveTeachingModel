import numpy as np
from . generic import Teacher


class Leitner(Teacher):

    def __init__(self, n_item, delay_factor, delay_min):

        box = np.full(n_item, -1)
        due = np.full(n_item, -1)

        self.n_item = n_item

        self.delay_factor = delay_factor
        self.delay_min = delay_min

        self.box = box
        self.due = due

    def update_box_and_due_time(self, last_idx,
                                last_was_success, last_time_reply):

        if last_was_success:
            self.box[last_idx] += 1
        else:
            self.box[last_idx] = \
                max(0, self.box[last_idx] - 1)

        delay = self.delay_factor ** self.box[last_idx]
        # Delay is 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 ... minutes
        self.due[last_idx] = \
            last_time_reply + self.delay_min * delay

    def _pickup_item(self, now):

        seen = np.argwhere(np.asarray(self.box) >= 0).flatten()
        n_seen = len(seen)

        if n_seen == self.n_item:
            return np.argmin(self.due)

        else:
            seen__due = np.asarray(self.due)[seen]
            seen__is_due = np.asarray(seen__due) <= now
            if np.sum(seen__is_due):
                seen_and_is_due__due = seen__due[seen__is_due]

                return seen[seen__is_due][np.argmin(seen_and_is_due__due)]
            else:
                return self._pickup_new()

    def _pickup_new(self):
        return np.argmin(self.box)

    def ask(self, now, last_was_success=None, last_time_reply=None,
            idx_last_q=None):

        if idx_last_q is None:
            item_idx = self._pickup_new()

        else:

            self.update_box_and_due_time(
                last_idx=idx_last_q,
                last_was_success=last_was_success,
                last_time_reply=last_time_reply)
            item_idx = self._pickup_item(now)

        return item_idx

    @classmethod
    def create(cls, tk, omniscient):
        return cls(n_item=tk.n_item,
                   delay_factor=tk.delay_factor,
                   delay_min=tk.delay_min)
