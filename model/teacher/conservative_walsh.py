import numpy as np

from model.learner.walsh2018 import Walsh2018


class ConservativeWalsh:

    def __init__(self, n_item, learnt_threshold, time_per_iter,
                 n_ss, ss_n_iter, time_between_ss):

        self.n_item = n_item
        self.thr = learnt_threshold

        self.eval_ts = n_ss * time_between_ss
        self.review_ts = np.hstack(
            [
                np.arange(x, x + (ss_n_iter * time_per_iter), time_per_iter)
                for x in np.arange(0, time_between_ss * n_ss, time_between_ss)
            ])

    @staticmethod
    def _p_seen(seen, hist, param, is_item_specific,
                ts, review_ts, cst_time):

        p_seen, seen = Walsh2018.p_seen_spec_hist(
            param=param, now=ts, hist=hist,
            ts=review_ts, seen=seen, is_item_specific=is_item_specific,
            cst_time=cst_time)
        return p_seen

    def _threshold_select(self, seen, param, hist, review_ts, n_item,
                          is_item_specific,
                          ts, cst_time):

        n_seen = np.sum(seen)

        if n_seen == 0:
            item = 0
        else:
            p_seen = self._p_seen(
                seen=seen,
                param=param,
                is_item_specific=is_item_specific,
                ts=ts,
                cst_time=cst_time,
                hist=hist,
                review_ts=review_ts)

            if n_seen == n_item or np.min(p_seen) <= self.thr:
                item = np.flatnonzero(seen)[np.argmin(p_seen)]
            else:
                item = np.argmin(seen)

        return item

    def _select(self, future_ts, review_ts, hist, param, eval_ts,
                cst_time, is_item_specific):

        n_item = self.n_item

        now = future_ts[0]
        future = future_ts[1:]

        hist_current = hist

        now_idx = np.flatnonzero(review_ts == now)[0]

        seen_current = np.zeros(n_item, dtype=bool)
        seen_current[np.unique(hist[hist != Walsh2018.DUMMY_VALUE])] = 1

        while True:

            seen = seen_current.copy()[:n_item]

            first_item = self._threshold_select(
                review_ts=review_ts,
                hist=hist,
                param=param,
                n_item=n_item,
                is_item_specific=is_item_specific,
                ts=now,
                cst_time=cst_time,
                seen=seen)

            n_item = first_item + 1

            hist = hist_current.copy()
            hist[now_idx] = first_item

            for i, ts in enumerate(future):

                item = self._threshold_select(
                    param=param,
                    n_item=n_item,
                    is_item_specific=is_item_specific,
                    ts=ts,
                    cst_time=cst_time,
                    hist=hist,
                    review_ts=review_ts,
                    seen=seen)

                seen[item] += 1
                hist[now_idx + 1 + i] = item

            p_seen = self._p_seen(
                seen=seen,
                param=param,
                is_item_specific=is_item_specific,
                ts=eval_ts,
                cst_time=cst_time,
                hist=hist,
                review_ts=review_ts)

            n_learnt = np.sum(p_seen > self.thr)
            if n_learnt == n_item:
                break

            n_item = first_item
            if n_item <= 1:
                break

        return first_item

    def ask(self, psy):

        param = psy.inferred_learner_param()
        hist = psy.learner.hist.copy()
        cst_time = psy.cst_time
        learner_model = psy.learner.__class__
        is_item_specific = psy.is_item_specific

        if learner_model != Walsh2018:
            raise ValueError

        no_dummy = hist != Walsh2018.DUMMY_VALUE
        current_step = np.sum(no_dummy)
        future_ts = self.review_ts[current_step:]

        item = self._select(
            is_item_specific=is_item_specific,
            future_ts=future_ts,
            cst_time=cst_time,
            eval_ts=self.eval_ts,
            param=param,
            hist=hist,
            review_ts=self.review_ts)

        return item
