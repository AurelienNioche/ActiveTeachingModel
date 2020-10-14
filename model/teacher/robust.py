import numpy as np

from model.learner.exponential import Exponential


class Robust:

    def __init__(self, n_item, learnt_threshold, time_per_iter,
                 n_ss, ss_n_iter, time_between_ss, n_sample=20):

        self.n_sample = n_sample

        self.n_item = n_item
        self.log_thr = np.log(learnt_threshold)

        self.eval_ts = n_ss * time_between_ss
        self.review_ts = np.hstack(
            [
                np.arange(x, x + (ss_n_iter * time_per_iter), time_per_iter)
                for x in np.arange(0, time_between_ss * n_ss, time_between_ss)
            ])

    def _threshold_select(self, n_pres, param, n_item, is_item_specific,
                          ts, last_pres, cst_time):

        if np.max(n_pres) == 0:
            item = 0
        else:
            seen = n_pres > 0

            log_p_seen = self._cp_log_p_seen(
                seen=seen,
                n_pres=n_pres,
                param=param,
                n_item=n_item,
                is_item_specific=is_item_specific,
                last_pres=last_pres,
                ts=ts,
                cst_time=cst_time)

            if np.sum(seen) == n_item or np.min(log_p_seen) <= self.log_thr:
                item = np.flatnonzero(seen)[np.argmin(log_p_seen)]
            else:
                item = np.argmin(seen)

        return item

    @staticmethod
    def _cp_log_p_seen(seen, n_pres, param, n_item, is_item_specific,
                       last_pres, ts, cst_time):

        if is_item_specific:
            init_forget = param[:n_item][seen, 0]
            rep_effect = param[:n_item][seen, 1]
        else:
            init_forget, rep_effect = param

        return \
            -init_forget \
            * (1 - rep_effect) ** (n_pres[seen] - 1) \
            * (ts - last_pres[seen]) \
            * cst_time

    def _recursive_exp_decay(self, n_pres, last_pres,
                             future_ts, param, eval_ts,
                             cst_time, is_item_specific):

        n_item = self.n_item

        now = future_ts[0]
        future = future_ts[1:]

        n_pres_current = n_pres
        last_pres_current = last_pres

        while True:

            n_pres = n_pres_current[:n_item]
            last_pres = last_pres_current[:n_item]

            first_item = self._threshold_select(
                n_pres=n_pres,
                param=param,
                n_item=n_item,
                is_item_specific=is_item_specific,
                ts=now, last_pres=last_pres,
                cst_time=cst_time)

            n_item = first_item + 1

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
                    cst_time=cst_time)

                n_pres[item] += 1
                last_pres[item] = ts

            seen = n_pres > 0
            log_p_seen = self._cp_log_p_seen(
                seen=seen,
                n_pres=n_pres,
                param=param,
                n_item=n_item,
                is_item_specific=is_item_specific,
                last_pres=last_pres,
                ts=eval_ts,
                cst_time=cst_time)

            n_learnt = np.sum(log_p_seen > self.log_thr)
            if n_learnt == n_item:
                break

            n_item = first_item
            if n_item <= 1:
                break

        return first_item, n_learnt

    def create_param_samples(self, log_post,
                             is_item_specific,
                             grid_param,
                             n_sample):

        post = np.exp(log_post)
        n_param_set, n_param = grid_param.shape

        if is_item_specific:
            n_item, n_param_set = log_post.shape
            param_list = np.zeros((n_sample, n_item, n_param))
            weights = np.zeros((n_sample, n_item))
            for i in range(n_item):
                slc = np.random.choice(np.arange(n_param_set),
                                       p=post[i], size=n_sample)
                param_list[:, i, :] = grid_param[slc]
                weights[:, i] = post[i, slc]
            weights = np.mean(weights, axis=1)
        else:
            slc = np.random.choice(np.arange(n_param_set),
                                   p=post, size=n_sample)
            param_list = grid_param[slc]
            weights = post[slc]

        return param_list, weights

    def get_future_timestamp_n_pres_last_pres(self, hist):

        no_dummy = hist != Exponential.DUMMY_VALUE
        current_step = np.sum(no_dummy)
        current_seen_item = np.unique(hist[no_dummy])

        future_ts = self.review_ts[current_step:]

        n_pres = np.zeros(self.n_item)
        last_pres = np.zeros(self.n_item)
        for i, item in enumerate(current_seen_item):
            is_item = hist == item
            n_pres[item] = np.sum(is_item)
            last_pres[item] = np.max(self.review_ts[is_item])

        return future_ts, n_pres, last_pres

    def ask(self, psy):

        hist = psy.learner.hist.copy()
        cst_time = psy.cst_time
        learner_model = psy.learner.__class__
        is_item_specific = psy.is_item_specific
        omniscient = psy.omniscient

        assert learner_model == Exponential

        future_ts, n_pres, last_pres = \
            self.get_future_timestamp_n_pres_last_pres(hist)

        if omniscient:

            param = psy.inferred_learner_param()
            item, expected_n_learnt = self._recursive_exp_decay(
                is_item_specific=is_item_specific,
                future_ts=future_ts,
                cst_time=cst_time,
                eval_ts=self.eval_ts,
                param=param,
                n_pres=n_pres,
                last_pres=last_pres)
        else:
            log_post = psy.log_post
            grid_param = psy.grid_param

            param_list, p_param_list = self.create_param_samples(
                log_post=log_post,
                is_item_specific=is_item_specific,
                grid_param=grid_param,
                n_sample=self.n_sample)
            rewards = np.zeros(self.n_sample)
            items = np.zeros(self.n_sample, dtype=int)
            for i, param in enumerate(param_list):
                items[i], rewards[i] = self._recursive_exp_decay(
                    is_item_specific=is_item_specific,
                    future_ts=future_ts,
                    cst_time=cst_time,
                    eval_ts=self.eval_ts,
                    param=param,
                    n_pres=n_pres,
                    last_pres=last_pres)
            # rewards *= p_param_list
            best = np.argmax(rewards)
            item = items[best]

        return item
