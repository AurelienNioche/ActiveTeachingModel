from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.utils import timezone

import numpy as np
import pandas as pd

from . psychologist import Psychologist, Learner

from . mcts_tools.mcts import MCTS


class LearnerStateParam:

    def __init__(self, learner_param, is_item_specific, learnt_threshold,
                 n_item, horizon, delta_timestep):

        self.horizon = horizon
        self.delta_timestep = delta_timestep

        self.n_item = n_item
        self.learnt_threshold = learnt_threshold

        self.learner_param = learner_param
        self.is_item_specific = is_item_specific


class LearnerState:

    def __init__(self, param, delta, n_pres, timestep):

        self.param = param

        self.timestep = timestep

        self.n_pres = n_pres
        self.delta = delta

        p_seen, seen = Learner.p_seen(
                param=self.param.learner_param,
                is_item_specific=self.param.is_item_specific,
                n_pres=n_pres,
                delta=delta
        )
        n_seen = np.sum(seen)

        self.reward = self._cpt_reward(p_seen)
        self.possible_actions = self._cpt_possible_actions(n_seen=n_seen)
        self.is_terminal = timestep == self.param.horizon
        self.rollout_action = self._cpt_rollout_action(n_seen=n_seen,
                                                       p_seen=p_seen)

        self.children = dict()

    def take_action(self, action):
        """Returns the state which results from taking action 'action'"""

        if action in self.children:
            new_state = self.children[action]

        else:
            n_pres = self.n_pres.copy()
            n_pres[action] += 1
            delta = self.delta.copy()
            dt = self.param.delta_timestep[self.timestep]
            delta += dt
            delta[action] = dt
            timestep = self.timestep + 1
            new_state = LearnerState(
                param=self.param,
                delta=delta,
                n_pres=n_pres,
                timestep=timestep,
            )
            self.children[action] = new_state

        return new_state

    def _cpt_rollout_action(self, p_seen, n_seen):

        if n_seen:
            n_item = self.param.n_item
            tau = self.param.learnt_threshold

            min_p = p_seen.min()

            if n_seen == n_item or min_p <= tau:
                item = p_seen.argmin()
            else:
                item = n_seen
        else:
            item = 0

        return item

    def _cpt_reward(self, p_seen):
        return np.mean(p_seen > self.param.learnt_threshold)

    def _cpt_possible_actions(self, n_seen):
        if n_seen == self.param.n_item:
            possible_actions = np.arange(self.param.n_item)
        else:
            possible_actions = np.arange(n_seen+1)
        return possible_actions


class MCTSTeacher:

    def __init__(self, n_item, learnt_threshold, bounds, grid_size,
                 is_item_specific, iter_limit, time_limit, horizon,
                 time_per_iter, ss_n_iter, ss_n_iter_between):

        self.psychologist = Psychologist(
            n_item=n_item,
            bounds=bounds,
            grid_size=grid_size,
            is_item_specific=is_item_specific)

        self.n_item = n_item

        self.learnt_threshold = learnt_threshold

        self.iter_limit = iter_limit
        self.time_limit = time_limit
        self.time_per_iter = time_per_iter
        self.horizon = horizon

        self.ss_n_iter = ss_n_iter
        self.ss_n_iter_between = ss_n_iter_between

        self.iter = 0
        self.ss_it = 0

    def _revise_goal(self):

        remain = self.ss_n_iter - self.ss_it

        print("iter", self.iter)
        print("horizon", self.horizon)

        if self.iter < self.horizon:
            h = self.horizon - self.iter
        else:
            self.iter = 0
            h = self.horizon

        print("iter", self.iter)
        print("h", h)

        if remain > h:
            delta_timestep = np.ones(h, dtype=int) * self.time_per_iter
        else:
            delta_timestep = np.zeros(h, dtype=int)
            delta_timestep[:remain] = \
                np.ones(remain) * self.time_per_iter
            next_ss = h - remain
            delta_timestep[remain:] = \
                (self.ss_n_iter_between + np.arange(next_ss)) \
                * self.time_per_iter

            assert h - remain <= self.ss_n_iter, "case not handled!"

        self.iter += 1
        self.ss_it += 1
        if self.ss_it == self.ss_n_iter:
            self.ss_it = 0
        return h, delta_timestep

    def _select_item(self, now):

        m = MCTS(iteration_limit=self.iter_limit, time_limit=self.time_limit)

        horizon, delta_timestep = self._revise_goal()

        n_pres = np.asarray(self.psychologist.n_pres)
        seen = n_pres >= 1
        last_pres = self.psychologist.last_pres[seen]
        sc = now - last_pres
        delta = np.full(self.n_item, -1, dtype=int)
        delta[seen] = sc
        learner_state = LearnerState(
            param=LearnerStateParam(
                learner_param=self.psychologist.inferred_learner_param(),
                is_item_specific=self.psychologist.is_item_specific,
                learnt_threshold=self.learnt_threshold,
                n_item=self.n_item,
                horizon=horizon,
                delta_timestep=delta_timestep,
            ),
            n_pres=n_pres,
            delta=delta,
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
