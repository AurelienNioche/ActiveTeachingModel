import numpy as np
from tqdm import tqdm
import datetime


class Leitner:

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

    def ask(self, now, last_was_success, last_time_reply, idx_last_q):

        if idx_last_q is None:
            item_idx = self._pickup_new()

        else:

            self.update_box_and_due_time(
                last_idx=idx_last_q,
                last_was_success=last_was_success,
                last_time_reply=last_time_reply)
            item_idx = self._pickup_item(now)

        return item_idx


def use_leitner(n_item, review_ts, rd, alpha, beta, eval_ts, thr):

    lei = Leitner(n_item=n_item, delay_min=2, delay_factor=2)

    n_pres = np.zeros(n_item, dtype=int)
    last_pres = np.zeros(n_item)

    last_item = None
    last_success = None
    last_ts = None

    for i, ts in enumerate(review_ts):

        item = lei.ask(now=ts, idx_last_q=last_item,
                       last_was_success=last_success,
                       last_time_reply=last_ts)
        p = 0 if not n_pres[item] else np.exp(-alpha
                                              * (1 - beta)**(n_pres[item] - 1)
                                              * (ts-last_pres[item]))
        last_success = p > rd[i]

        n_pres[item] += 1
        last_pres[item] = ts

        last_ts = ts
        last_item = item

    seen = n_pres > 0
    p_seen = np.exp(-alpha
                    * (1 - beta) ** (n_pres[seen] - 1)
                    * (eval_ts-last_pres[seen]))
    n_learnt = np.sum(p_seen > thr)
    print("leitner n learnt", n_learnt)
    print()


def use_threshold(n_item, alpha, beta, review_ts, eval_ts, thr):

    n_pres = np.zeros(n_item, dtype=int)
    last_pres = np.zeros(n_item)

    for i, ts in enumerate(review_ts):

        if np.max(n_pres) == 0:
            item = 0
        else:
            seen = n_pres > 0
            p_seen = np.exp(-alpha * (1 - beta) ** (n_pres[seen] - 1) * (
                    ts - last_pres[seen]))
            if np.min(p_seen) <= 0.90:
                item = np.flatnonzero(seen)[np.argmin(p_seen)]
            else:
                item = np.min([n_item, np.max(np.flatnonzero(seen)) + 1])

        n_pres[item] += 1
        last_pres[item] = ts

    seen = n_pres > 0
    p_seen = np.exp(-alpha * (1 - beta) ** (n_pres[seen] - 1) * (
            eval_ts - last_pres[seen]))

    n_learnt = np.sum(p_seen > thr)
    print("threshold n learnt", n_learnt)
    print()


def build_hist(n_item, n_total_iter):
    hist = [0]
    for i in range(n_total_iter-1):
        hist.append(np.random.randint(0,
                                      min(n_item, np.unique(hist).size + 1)))

    return hist
#     return np.random.choice(np.arange(n_item),
#                             replace=True,
#                             size=n_total_iter)


def use_random_sampling(n_item, review_ts, alpha, beta, eval_ts, thr):

    n_total_iter = len(review_ts)

    n_sample = 500
    n_learnt = np.zeros(n_sample, dtype=int)

    for s in tqdm(range(n_sample)):

        hist = build_hist(n_item, n_total_iter)

        seen_item = np.unique(hist)
        n_pres = np.zeros(n_item)
        last_pres = np.zeros(n_item)
        for i, item in enumerate(seen_item):
            is_item = hist == item
            n_pres[i] = np.sum(is_item)
            last_pres[i] = np.max(review_ts[is_item])

        seen = n_pres > 0
        p_seen = np.exp(-alpha * (1 - beta) ** (n_pres[seen] - 1) * (
                    eval_ts - last_pres[seen]))
        n_learnt[s] = np.sum(p_seen > thr)

    print("random sampling n learnt", np.max(n_learnt))
    # plt.hist(n_learnt)
    # plt.show()


def recursive(n_item, review_ts, alpha, beta, eval_ts, thr, verbose=False):

    old_n_learnt = 0
    itr = 0
    n_item = n_item
    while True:

        n_pres = np.zeros(n_item, dtype=int)
        last_pres = np.zeros(n_item)

        for i, ts in enumerate(review_ts):

            if np.max(n_pres) == 0:
                item = 0
            else:
                seen = n_pres > 0
                p_seen = np.exp(-alpha * (1 - beta) ** (n_pres[seen] - 1) * (
                        ts - last_pres[seen]))
                if np.min(p_seen) <= 0.90 or np.sum(seen) == n_item:
                    item = np.flatnonzero(seen)[np.argmin(p_seen)]
                else:
                    item = np.max(np.flatnonzero(seen)) + 1

            n_pres[item] += 1
            last_pres[item] = ts

        seen = n_pres > 0
        p_seen = np.exp(-alpha * (1 - beta) ** (n_pres[seen] - 1) * (
                    eval_ts - last_pres[seen]))

        n_learnt = np.sum(p_seen > thr)
        if verbose:
            print("--- iter", itr, "n item", n_item,
                  "n learnt", n_learnt, "---")

        if n_learnt == n_item:
            old_n_learnt = n_item
            break
        elif n_learnt < old_n_learnt:
            break
        else:
            old_n_learnt = n_learnt
            n_item = np.sum(n_pres > 0) - 1
            itr += 1
            continue

    print("recursive n learnt", old_n_learnt)
    print()


def main():

    alpha, beta = [0.02 * 0.003, 0.44]
    ss_n_iter = 100
    time_per_iter = 2
    n_sec_day = 24*60**2
    n_ss = 6
    eval_ts = n_ss*n_sec_day
    review_ts = np.hstack([np.arange(x,
                                     x+(ss_n_iter*time_per_iter),
                                     time_per_iter)
                           for x in np.arange(0, n_sec_day*n_ss, n_sec_day)])

    n_item = 150

    thr = 0.90

    n_total_iter = len(review_ts)

    np.random.seed(123)
    rd = np.random.random(size=n_total_iter)

    use_leitner(n_item=n_item,
                review_ts=review_ts, eval_ts=eval_ts,
                rd=rd,
                alpha=alpha, beta=beta,
                thr=thr)

    use_threshold(n_item=n_item,
                  alpha=alpha,
                  beta=beta,
                  review_ts=review_ts,
                  eval_ts=eval_ts, thr=thr)
    a = datetime.datetime.now()
    recursive(n_item=n_item,
              review_ts=review_ts,
              alpha=alpha,
              beta=beta, eval_ts=eval_ts, thr=thr)
    print(datetime.datetime.now() - a)

    use_random_sampling(n_item=n_item,
                        review_ts=review_ts,
                        alpha=alpha,
                        beta=beta,
                        eval_ts=eval_ts,
                        thr=thr)


if __name__ == "__main__":
    main()
