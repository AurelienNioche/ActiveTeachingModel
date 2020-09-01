import numpy as np
import itertools as it
from . generic import Teacher

from model.learner.exponential_n_delta import ExponentialNDelta
from model.learner.walsh2018 import Walsh2018


class SamplingEnd(Teacher):

    def __init__(self, n_item, learnt_threshold,
                 n_sample,
                 time_per_iter, n_ss, ss_n_iter, time_between_ss):

        self.n_sample = n_sample

        self.n_item = n_item
        self.learnt_threshold = learnt_threshold
        self.time_per_iter = time_per_iter
        self.ss_n_iter = ss_n_iter
        self.time_between_ss = time_between_ss

        # Time between each *beginning* of session
        time_between_ss -= time_per_iter * ss_n_iter

        eval_ts = n_ss * time_between_ss
        review_ts = np.hstack([np.arange(x,
                                         x + (ss_n_iter * time_per_iter),
                                         time_per_iter)
                               for x in
                               np.arange(0, time_between_ss * n_ss,
                                         time_between_ss)])

        self.eval_ts = eval_ts

    def _recursive_exp_decay(self, review_ts, param, eval_ts, thr):

        alpha, beta = param

        old_n_learnt = 0
        itr = 0
        n_item = self.n_item
        while True:

            n_pres = np.zeros(n_item, dtype=int)
            last_pres = np.zeros(n_item)

            for i, ts in enumerate(review_ts):

                if np.max(n_pres) == 0:
                    item = 0
                else:
                    seen = n_pres > 0
                    p_seen = np.exp(
                        -alpha * (1 - beta) ** (n_pres[seen] - 1) * (
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
            print("--- iter", itr, "n item", n_item, "n learnt", n_learnt,
                  "---")

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

    def _recursive_walsh(self, review_ts, learner_model,
                         hist, param, cst_time,
                         is_item_specific, eval_ts):

        old_n_learnt = 0
        itr = 0
        n_item = self.n_item

        current_step = np.sum(hist != 1)
        seen_item = np.delete(np.unique(hist), -1)

        while True:

            seen = np.zeros(n_item, dtype=int)
            seen[seen_item[seen_item < n_item]] = True
            hist[current_step:] = -1

            for i, ts in enumerate(review_ts[current_step:]):

                if np.sum(seen) == 0:
                    item = 0
                else:

                    p_seen = learner_model.p_seen_spec_hist(
                        param=param, now=ts,
                        hist=hist, ts=review_ts,
                        seen=seen, is_item_specific=is_item_specific,
                        cst_time=cst_time)

                    if np.min(p_seen) <= self.learnt_threshold \
                            or np.sum(seen) == n_item:
                        item = np.flatnonzero(seen)[np.argmin(p_seen)]
                    else:
                        item = np.max(np.flatnonzero(seen)) + 1

                hist[current_step + i] = item
                seen[item] = True

            p_seen = learner_model.p_seen_spec_hist(
                param=param, now=eval_ts,
                hist=hist, ts=review_ts,
                seen=seen, is_item_specific=is_item_specific,
                cst_time=cst_time)

            n_learnt = np.sum(p_seen > self.learnt_threshold)
            print("--- iter", itr, "n item", n_item, "n learnt", n_learnt,
                  "---")

            if n_learnt == n_item:
                old_n_learnt = n_item
                break
            elif n_learnt < old_n_learnt:
                break
            else:
                old_n_learnt = n_learnt
                n_item = np.sum(seen) - 1
                itr += 1
                continue

        print("recursive n learnt", old_n_learnt)
        print()

    def _revise_goal(self, now, ss_iter):

        h = self.ss_n_iter - ss_iter
        ts = now + np.arange(h) * self.time_per_iter
        eval_ts = ts[-1] + self.time_between_ss
        return h, ts, eval_ts

    def _value_future(self, psy, future, param, new_ts, eval_ts):
        hist = psy.learner.hist
        hist = hist[hist != -1]
        new_hist = \
            np.hstack((hist, future))

        seen = np.zeros(self.n_item, dtype=bool)
        seen[np.unique(new_hist)] = True

        p_seen, seen = psy.learner.p_seen_spec_hist(
            param=param, hist=new_hist,
            ts=new_ts, seen=seen,
            now=eval_ts,
            is_item_specific=psy.is_item_specific
        )
        return np.sum(p_seen > self.learnt_threshold), np.sum(p_seen)

    def ask(self, now, psy, eval_ts):

        param = psy.inferred_learner_param()
        ts = psy.learner.ts
        hist = psy.learner.hist

        if isinstance(psy.learner, ExponentialNDelta):
            item =


        return item_idx

