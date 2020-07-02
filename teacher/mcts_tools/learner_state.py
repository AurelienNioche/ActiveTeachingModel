import numpy as np
from copy import deepcopy


class StateParam:

    def __init__(self, learner_param, is_item_specific, learnt_threshold,
                 n_item, horizon, timestamps):

        self.horizon = horizon
        self.timestamps = timestamps

        self.n_item = n_item
        self.learnt_threshold = learnt_threshold

        self.learner_param = learner_param
        self.is_item_specific = is_item_specific


class LearnerState:

    def __init__(self, param, learner, timestep):

        self.param = param

        self.timestep = timestep

        self.learner = learner

        p_seen, seen = learner.p_seen(
                param=self.param.learner_param,
                is_item_specific=self.param.is_item_specific,
                now=self.param.timestamps[timestep]
        )
        n_seen = np.sum(seen)

        self.reward = self._cpt_reward(p_seen)
        self.possible_actions = self._cpt_possible_actions(n_seen=n_seen)
        self.is_terminal = timestep == self.param.horizon
        self.rollout_action = self._cpt_rollout_action(n_seen=n_seen,
                                                       p_seen=p_seen)

        self.children = dict()

    def take_action(self, action):

        if action in self.children:
            new_state = self.children[action]

        else:
            new_learner = deepcopy(self.learner)
            new_learner.update(item=action,
                               timestamp=self.param.timestamps[self.timestep])
            new_timestep = self.timestep + 1
            new_state = LearnerState(
                param=self.param,
                learner=new_learner,
                timestep=new_timestep,
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
