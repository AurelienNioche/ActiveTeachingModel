import numpy as np
from copy import deepcopy

from . mcts_tools.mcts import MCTS
from . mcts_tools.learner_state import LearnerState, StateParam
from . generic import Teacher


class MCTSTeacher(Teacher):

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

        m = MCTS(iteration_limit=self.iter_limit, time_limit=self.time_limit)

        horizon, timestamps = self._revise_goal(now)

        param = self.psychologist.inferred_learner_param()
        learner_state = LearnerState(
            param=StateParam(
                learner_param=param,
                is_item_specific=self.psychologist.is_item_specific,
                learnt_threshold=self.learnt_threshold,
                n_item=self.n_item,
                horizon=horizon,
                timestamps=timestamps,
            ),
            learner=deepcopy(self.psychologist.learner),
            timestep=0
        )

        item_idx = m.run(initial_state=learner_state)
        return item_idx

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
