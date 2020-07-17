import numpy as np
import itertools as it
from . generic import Teacher


class Sampling(Teacher):

    def __init__(self, n_item, learnt_threshold,
                 iter_limit, time_limit, horizon,
                 time_per_iter, ss_n_iter, ss_n_iter_between,
                 psychologist):

        self.psychologist = psychologist

        self.n_item = n_item

        self.learnt_threshold = learnt_threshold

        self.iter_limit = iter_limit
        self.time_limit = time_limit
        self.time_per_iter = time_per_iter
        self.horizon = horizon

        self.ss_n_iter = ss_n_iter
        self.ss_n_iter_between = ss_n_iter_between

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
        delta_ts = np.arange(h + 1, dtype=int)

        if remain < h + 1:
            delta_ts[remain:] += self.ss_n_iter_between
            assert h - remain <= self.ss_n_iter, "case not handled!"

        timestamps = now + delta_ts * self.time_per_iter

        return h, timestamps

    def _select_item(self, now):
        # print()

        horizon, timestamps = self._revise_goal(now)
        # print("h", horizon, "ts", len(timestamps))
        param = self.psychologist.inferred_learner_param()
        ts = self.psychologist.learner.ts
        ts = ts[ts != -1]
        new_ts = \
            np.hstack((ts, timestamps[:-1]))
        eval_ts = timestamps[-1]

        items = np.hstack((self.psychologist.learner.seen_item,
                           [self.psychologist.learner.n_seen, ]))

        n_item = len(items)
        n_perm = n_item ** horizon

        if n_perm < self.iter_limit:

            r = np.zeros(n_perm)
            first = np.zeros(n_perm, dtype=int)
            for i, future in enumerate(it.product(items, repeat=horizon)):
                first[i] = future[0]
                r[i] = self._value_future(future=future, param=param,
                                          new_ts=new_ts, eval_ts=eval_ts)

            item_idx = first[np.argmax(r)]
        else:
            r = np.zeros(self.iter_limit)
            first = np.zeros(self.iter_limit, dtype=int)
            for i in range(self.iter_limit):
                future = np.random.choice(items, replace=True, size=horizon)
                first[i] = future[0]
                r[i] = self._value_future(future=future, param=param,
                                          new_ts=new_ts, eval_ts=eval_ts)

            item_idx = first[np.argmax(r)]

        return item_idx

    def _value_future(self, future, param, new_ts, eval_ts):
        hist = self.psychologist.learner.hist
        hist = hist[hist != -1]
        new_hist = \
            np.hstack((hist, future))

        seen = np.zeros(self.n_item, dtype=bool)
        seen[np.unique(new_hist)] = True

        p_seen, seen = self.psychologist.learner.p_seen_spec_hist(
            param=param, hist=new_hist,
            ts=new_ts, seen=seen,
            now=eval_ts
        )
        return np.sum(p_seen)

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

    @classmethod
    def create(cls, tk, omniscient):

        psychologist = tk.psychologist_model.create(
            tk=tk,
            omniscient=omniscient)
        return cls(
            psychologist=psychologist,
            n_item=tk.n_item,
            learnt_threshold=tk.learnt_threshold,
            iter_limit=tk.iter_limit,
            time_limit=tk.time_limit,
            horizon=tk.horizon,
            time_per_iter=tk.time_per_iter,
            ss_n_iter=tk.ss_n_iter,
            ss_n_iter_between=tk.ss_n_iter_between)
