import numpy as np
from . generic import Teacher

from model.learner.exponential_n_delta import ExponentialNDelta
# from model.learner.walsh2018 import Walsh2018
from tqdm import tqdm
import datetime


class Recursive(Teacher):

    def __init__(self, n_item, learnt_threshold,
                 time_per_iter, n_ss, ss_n_iter, time_between_ss):

        self.n_item = n_item
        self.learnt_threshold = learnt_threshold

        # Time between each *beginning* of session
        time_between_ss -= time_per_iter * ss_n_iter

        self.eval_ts = n_ss * time_between_ss
        self.review_ts = np.hstack([
            np.arange(x,
                      x + (ss_n_iter * time_per_iter),
                      time_per_iter)
            for x in np.arange(0,
                               time_between_ss * n_ss,
                               time_between_ss)])

    def _recursive_exp_decay(self, hist, review_ts, param, eval_ts,
                             cst_time, is_item_specific):

        itr = 0
        n_item = self.n_item

        thr = self.learnt_threshold

        no_dummy = hist != ExponentialNDelta.DUMMY_VALUE
        current_step = np.sum(no_dummy)
        current_seen_item = np.unique(hist[no_dummy])

        future = review_ts[current_step:]

        first_item = None

        n_pres_current = np.zeros(self.n_item)
        last_pres_current = np.zeros(self.n_item)
        for i, item in enumerate(current_seen_item):
            is_item = hist == item
            n_pres_current[item] = np.sum(is_item)
            last_pres_current[item] = np.max(review_ts[is_item])

        while True:

            n_pres = n_pres_current[:n_item].copy()
            last_pres = last_pres_current[:n_item].copy()

            for i, ts in enumerate(future):

                if np.max(n_pres) == 0:
                    item = 0
                else:
                    seen = n_pres > 0

                    if is_item_specific:
                        init_forget = param[seen, 0]
                        rep_effect = param[seen, 1]
                    else:
                        init_forget, rep_effect = param

                    p_seen = np.exp(
                        -init_forget
                        * (1 - rep_effect) ** (n_pres[seen] - 1)
                        * (ts - last_pres[seen])
                        * cst_time)

                    if np.min(p_seen) <= 0.90 or np.sum(seen) == n_item:
                        item = np.flatnonzero(seen)[np.argmin(p_seen)]
                    else:
                        item = np.max(np.flatnonzero(seen)) + 1

                if i == 0:
                    first_item = item

                n_pres[item] += 1
                last_pres[item] = ts

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
                * cst_time)

            n_learnt = np.sum(p_seen > thr)

            if n_learnt == n_item:
                break

            elif n_item <= 1:
                break
            else:
                n_item = np.sum(n_pres > 0) - 1

                itr += 1
                continue

        return first_item

    def _recursive(self, review_ts, learner_model,
                   hist, param, cst_time,
                   is_item_specific, eval_ts):

        itr = 0
        n_item = self.n_item
        lm = learner_model

        current_step = np.sum(hist != lm.DUMMY_VALUE)
        current_seen_item = np.unique(hist[hist != lm.DUMMY_VALUE])

        future = review_ts[current_step:]

        first_item = None

        while True:

            seen = np.zeros(n_item, dtype=int)
            seen[current_seen_item[current_seen_item < n_item]] = True
            hist[current_step:] = lm.DUMMY_VALUE

            # a = datetime.datetime.now()

            for i, ts in enumerate(future):

                if np.sum(seen) == 0:
                    item = 0
                    first_item = item
                else:

                    p_seen, _ = lm.p_seen_spec_hist(
                        param=param, now=ts,
                        hist=hist, ts=review_ts,
                        seen=seen, is_item_specific=is_item_specific,
                        cst_time=cst_time)
                    min_p_seen = np.min(p_seen)
                    n_seen = np.sum(seen)
                    item_seen = np.flatnonzero(seen)
                    if min_p_seen <= self.learnt_threshold \
                            or n_seen == n_item:
                        item = item_seen[np.argmin(p_seen)]
                    else:
                        item = np.max(item_seen) + 1
                    if i == 0:
                        first_item = item

                hist[current_step + i] = item
                seen[item] = True
            #
            # print(n_item, datetime.datetime.now() - a)

            p_seen, _ = lm.p_seen_spec_hist(
                param=param, now=eval_ts,
                hist=hist, ts=review_ts,
                seen=seen, is_item_specific=is_item_specific,
                cst_time=cst_time)

            n_learnt = np.sum(p_seen > self.learnt_threshold)

            if n_learnt == n_item:
                # old_n_learnt = n_item
                break

            elif n_item <= 1:
                break
            else:
                n_item = np.sum(seen) - 1

                itr += 1
                continue

        return first_item

    def ask(self, now, psy):

        param = psy.inferred_learner_param()
        hist = psy.learner.hist.copy()
        cst_time = psy.cst_time
        learner_model = psy.learner.__class__
        is_item_specific = psy.is_item_specific

        if learner_model == ExponentialNDelta:

            item = self._recursive_exp_decay(
                is_item_specific=is_item_specific,
                review_ts=self.review_ts,
                cst_time=cst_time,
                eval_ts=self.eval_ts,
                param=param,
                hist=hist)

        else:

            item = self._recursive(
                review_ts=self.review_ts,
                cst_time=cst_time,
                eval_ts=self.eval_ts,
                param=param,
                hist=hist,
                is_item_specific=is_item_specific,
                learner_model=learner_model)

        return item
