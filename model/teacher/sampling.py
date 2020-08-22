import numpy as np
import itertools as it
from . generic import Teacher


class Sampling(Teacher):

    def __init__(self, n_item, learnt_threshold,
                 n_sample, horizon,
                 time_per_iter, ss_n_iter, time_between_ss):

        self.n_item = n_item

        self.learnt_threshold = learnt_threshold

        self.n_sample = n_sample

        self.time_per_iter = time_per_iter
        self.horizon = horizon

        self.ss_n_iter = ss_n_iter

        self.time_between_ss = time_between_ss

        self.iter = -1
        self.ss_it = -1

    def _revise_goal(self, now):

        self.ss_it += 1
        if self.ss_it == self.ss_n_iter - 1:
            self.ss_it = 0

        remain = self.ss_n_iter - self.ss_it

        self.iter += 1
        if self.iter == self.horizon:
            self.iter = 0
            h = self.horizon
        else:
            h = self.horizon - self.iter

        # delta in timestep (number of iteration)
        delta_ts = np.arange(h + 1, dtype=float)

        if remain < h + 1:
            delta_ts[remain:] += (self.time_between_ss/self.time_per_iter)
            assert h - remain <= self.ss_n_iter, "case not handled!"

        timestamps = now + delta_ts * self.time_per_iter

        return h, timestamps

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
        return np.sum(p_seen)

    def ask(self, now, psy):

        horizon, timestamps = self._revise_goal(now)

        param = psy.inferred_learner_param()
        ts = psy.learner.ts
        ts = ts[ts != -1]
        new_ts = \
            np.hstack((ts, timestamps[:-1]))
        eval_ts = timestamps[-1]

        if psy.learner.n_seen < self.n_item:
            items = np.hstack((psy.learner.seen_item,
                               [psy.learner.n_seen, ]))
        else:
            items = np.arange(self.n_item)

        n_item = len(items)
        n_perm = n_item ** horizon

        if n_perm < self.n_sample:

            r = np.zeros(n_perm)
            first = np.zeros(n_perm, dtype=int)
            for i, future in enumerate(it.product(items, repeat=horizon)):
                first[i] = future[0]
                r[i] = self._value_future(psy=psy, future=future, param=param,
                                          new_ts=new_ts, eval_ts=eval_ts)

            item_idx = first[np.argmax(r)]

        else:
            r = np.zeros(self.n_sample)
            first = np.zeros(self.n_sample, dtype=int)
            for i in range(self.n_sample):
                future = np.random.choice(items, replace=True, size=horizon)
                first[i] = future[0]
                r[i] = self._value_future(psy=psy, future=future, param=param,
                                          new_ts=new_ts, eval_ts=eval_ts)

            item_idx = first[np.argmax(r)]

        return item_idx

    @classmethod
    def create(cls, n_item, task_pr, n_sample, horizon):

        return cls(
            n_item=n_item,
            learnt_threshold=task_pr.learnt_threshold,
            n_sample=n_sample,
            horizon=horizon,
            time_per_iter=task_pr.time_per_iter,
            ss_n_iter=task_pr.ss_n_iter,
            time_between_ss=task_pr.time_between_ss)
