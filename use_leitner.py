import datetime

import numpy as np
from tqdm import tqdm


class Leitner:
    def __init__(self, n_item, delay_factor, delay_min):

        box = np.full(n_item, -1)
        due = np.full(n_item, -1)

        self.n_item = n_item

        self.delay_factor = delay_factor
        self.delay_min = delay_min

        self.box = box
        self.due = due

    def update_box_and_due_time(self, last_idx, last_was_success, last_time_reply):

        if last_was_success:
            self.box[last_idx] += 1
        else:
            self.box[last_idx] = max(0, self.box[last_idx] - 1)

        delay = self.delay_factor ** self.box[last_idx]
        # Delay is 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 ... minutes
        self.due[last_idx] = last_time_reply + self.delay_min * delay

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

    def ask(self, now, last_was_success, last_time_reply, idx_last_q):

        if idx_last_q is None:
            item_idx = self._pickup_new()

        else:

            self.update_box_and_due_time(
                last_idx=idx_last_q,
                last_was_success=last_was_success,
                last_time_reply=last_time_reply,
            )
            item_idx = self._pickup_item(now)

        return item_idx


def use_leitner(n_item, review_ts, rd, param, eval_ts, thr):

    is_item_specific = len(np.asarray(param).shape) > 1

    lei = Leitner(n_item=n_item, delay_min=2, delay_factor=2)

    n_pres = np.zeros(n_item, dtype=int)
    last_pres = np.zeros(n_item)

    last_item = None
    last_success = None
    last_ts = None

    for i, ts in enumerate(review_ts):

        item = lei.ask(
            now=ts,
            idx_last_q=last_item,
            last_was_success=last_success,
            last_time_reply=last_ts,
        )

        if is_item_specific:
            init_forget = param[item, 0]
            rep_effect = param[item, 1]
        else:
            init_forget, rep_effect = param

        p = (
            0
            if not n_pres[item]
            else np.exp(
                -init_forget
                * (1 - rep_effect) ** (n_pres[item] - 1)
                * (ts - last_pres[item])
            )
        )

        last_success = p > rd[i]

        n_pres[item] += 1
        last_pres[item] = ts

        last_ts = ts
        last_item = item

    seen = n_pres > 0

    if is_item_specific:
        init_forget = param[seen, 0]
        rep_effect = param[seen, 1]
    else:
        init_forget, rep_effect = param

    p_seen = np.exp(
        -init_forget
        * (1 - rep_effect) ** (n_pres[seen] - 1)
        * (eval_ts - last_pres[seen])
    )
    n_learnt = np.sum(p_seen > thr)
    print("leitner n learnt", n_learnt)
    print()


def main():

    n_item = 150

    param = [0.0006, 0.44] # [[0.0006, 0.44] for _ in range(n_item)]
    param = np.asarray(param)

    ss_n_iter = 100
    time_per_iter = 4
    n_sec_day = 24 * 60 ** 2
    n_ss = 6
    eval_ts = n_ss * n_sec_day
    review_ts = np.hstack([
            np.arange(x, x + (ss_n_iter * time_per_iter), time_per_iter)
            for x in np.arange(0, n_sec_day * n_ss, n_sec_day)
        ])

    thr = 0.90

    n_total_iter = len(review_ts)

    np.random.seed(123)
    rd = np.random.random(size=n_total_iter)

    a = datetime.datetime.now()

    use_leitner(
        n_item=n_item, review_ts=review_ts, eval_ts=eval_ts,
        rd=rd, param=param, thr=thr)

    print("[Time to execute ", datetime.datetime.now() - a, "]")


if __name__ == "__main__":
    main()
