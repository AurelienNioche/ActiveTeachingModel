import numpy as np

from model.learner.exponential_n_delta import ExponentialNDelta
from .generic import Teacher


class RecursiveInverse(Teacher):

    def __init__(self, n_item, learnt_threshold, time_per_iter,
                 n_ss, ss_n_iter, time_between_ss):

        self.n_item = n_item
        self.learnt_threshold = learnt_threshold

        # Time between each *beginning* of session
        # time_between_ss += time_per_iter * ss_n_iter

        self.eval_ts = n_ss * time_between_ss
        self.review_ts = np.hstack(
            [
                np.arange(x, x + (ss_n_iter * time_per_iter), time_per_iter)
                for x in np.arange(0, time_between_ss * n_ss, time_between_ss)
            ])

    @staticmethod
    def _threshold_select(n_pres, param, n_item, is_item_specific,
                          ts, last_pres, cst_time, thr):

        if np.max(n_pres) == 0:
            item = 0
        else:
            seen = n_pres > 0

            if is_item_specific:
                init_forget = param[:n_item][seen, 0]
                rep_effect = param[:n_item][seen, 1]
            else:
                init_forget, rep_effect = param

            p_seen = np.exp(
                -init_forget
                * (1 - rep_effect) ** (n_pres[seen] - 1)
                * (ts - last_pres[seen])
                * cst_time)

            if np.min(p_seen) <= thr or np.sum(seen) == n_item:
                item = np.flatnonzero(seen)[np.argmin(p_seen)]
            else:
                item = np.max(np.flatnonzero(seen)) + 1

        return item

    def _recursive_exp_decay(self, hist, review_ts, param, eval_ts,
                             cst_time, is_item_specific):

        itr = 0
        n_item = self.n_item

        thr = self.learnt_threshold

        no_dummy = hist != ExponentialNDelta.DUMMY_VALUE
        current_step = np.sum(no_dummy)
        current_seen_item = np.unique(hist[no_dummy])

        now = review_ts[current_step]
        future = review_ts[current_step+1:]

        n_pres_current = np.zeros(self.n_item)
        last_pres_current = np.zeros(self.n_item)
        for i, item in enumerate(current_seen_item):
            is_item = hist == item
            n_pres_current[item] = np.sum(is_item)
            last_pres_current[item] = np.max(review_ts[is_item])

        while True:

            n_pres = n_pres_current[:n_item].copy()
            last_pres = last_pres_current[:n_item].copy()

            first_item = self._threshold_select(
                n_pres=n_pres,
                param=param,
                n_item=n_item,
                is_item_specific=is_item_specific,
                ts=now, last_pres=last_pres,
                cst_time=cst_time,
                thr=thr)

            n_item = min(first_item + 1, self.n_item)

            n_pres = n_pres_current[:n_item].copy()
            last_pres = last_pres_current[:n_item].copy()

            n_pres[first_item] += 1
            last_pres[first_item] = now

            for ts in future:

                item = self._threshold_select(
                    n_pres=n_pres,
                    param=param,
                    n_item=n_item,
                    is_item_specific=is_item_specific,
                    ts=ts, last_pres=last_pres,
                    cst_time=cst_time,
                    thr=thr)

                n_pres[item] += 1
                last_pres[item] = ts

            # if is_item_specific:
            #     init_forget = param[:n_item][first_item, 0]
            #     rep_effect = param[:n_item][first_item, 1]
            # else:
            #     init_forget, rep_effect = param
            #
            # p = np.exp(
            #     -init_forget
            #     * (1 - rep_effect) ** (n_pres[first_item] - 1)
            #     * (eval_ts - last_pres[first_item])
            #     * cst_time)
            #
            # learnt = p > thr

            seen = n_pres > 0

            if is_item_specific:
                init_forget = param[:n_item][seen, 0]
                rep_effect = param[:n_item][seen, 1]
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

            n_item = first_item

            if n_item <= 1:
                break

            itr += 1

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
            raise NotImplementedError

        return item
