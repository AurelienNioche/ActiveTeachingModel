import numpy as np
from copy import deepcopy

from . mcts_tools.mcts import MCTS
from . mcts_tools.learner_state import LearnerState, StateParam
from . generic import Teacher


class MCTSTeacher(Teacher):

    def __init__(self, n_item, learnt_threshold,
                 iter_limit, time_limit, horizon,
                 time_per_iter, ss_n_iter, time_between_ss):

        self.n_item = n_item

        self.learnt_threshold = learnt_threshold

        self.iter_limit = iter_limit
        self.time_limit = time_limit
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
        delta_ts = np.arange(h + 1, dtype=int)

        if remain < h + 1:
            delta_ts[remain:] += (self.time_between_ss / self.time_per_iter)
            assert h - remain <= self.ss_n_iter, "case not handled!"

        timestamps = now + delta_ts * self.time_per_iter

        return h, timestamps

    def _select_item(self, psy, now):

        m = MCTS(iteration_limit=self.iter_limit, time_limit=self.time_limit)

        horizon, timestamps = self._revise_goal(now)

        param = psy.inferred_learner_param()
        learner_state = LearnerState(
            param=StateParam(
                learner_param=param,
                is_item_specific=psy.is_item_specific,
                learnt_threshold=self.learnt_threshold,
                n_item=self.n_item,
                horizon=horizon,
                timestamps=timestamps,
            ),
            learner=deepcopy(psy.learner),
            timestep=0
        )

        item_idx = m.run(initial_state=learner_state)
        return item_idx

    def ask(self, now, psy, last_was_success, last_time_reply, idx_last_q):

        if idx_last_q is None:
            return 0
        else:
            return self._select_item(now=now, psy=psy)

    @classmethod
    def create(cls, n_item, task_pr,
               iter_limit, time_limit, horizon):

        return cls(
            n_item=n_item,
            learnt_threshold=task_pr.learnt_threshold,
            time_per_iter=task_pr.time_per_iter,
            time_between_ss=task_pr.time_between,
            ss_n_iter=task_pr.ss_n_iter,
            iter_limit=iter_limit,
            time_limit=time_limit,
            horizon=horizon)
