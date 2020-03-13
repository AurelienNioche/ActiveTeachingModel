"""
Adapted from: https://github.com/pbsinclair42/MCTS/blob/master/mcts.py
"""
import numpy as np


class State:

    def get_possible_actions(self):
        """Returns an iterable of all actions which can be taken
        from this state"""

    def take_action(self, action):
        """Returns the state which results from taking action 'action'"""

    def is_terminal(self, ):
        """Returns whether this state is a terminal state"""

    def get_reward(self):
        """"Returns the reward for this state"""


class BasicLearnerState(State):

    def __init__(self, seen, n_item):

        self.n_item = n_item
        self.seen = seen

    def get_possible_actions(self):
        """Returns an iterable of all actions which can be taken
        from this state"""
        return np.arange(self.n_item)

    def take_action(self, action):
        """Returns the state which results from taking action 'action'"""
        new_seen = self.seen.copy()
        new_seen[action] = 1
        return self.__class__(new_seen)

    def is_terminal(self):
        """Returns whether this state is a terminal state"""
        return False

    def get_reward(self):
        """"Returns the reward for this state"""
        return np.sum(self.seen == 1)

    def __str__(self):
        return f"State: {self.seen}"


class LearnerState(State):

    def __init__(self, n_pres, delta,
                 learnt_thr,
                 horizon,
                 param,
                 root_action=None,
                 parent=None):

        self.n_pres = n_pres
        self.delta = delta

        self.param = param
        self.learnt_thr = learnt_thr
        self.horizon = horizon

        self.n_item = self.n_pres.size

        self._instant_reward = None
        self._possible_actions = None

        self.parent = parent
        self.children = dict()

        self.root_action = root_action

        self.t = None
        self.cumulative_reward = None
        self.parent = None

    def get_possible_actions(self):
        """Returns an iterable of all actions which can be taken
        from this state"""
        if self._possible_actions is None:
            seen = self.n_pres[:] > 0
            n_seen = np.sum(seen)
            if n_seen == 0:
                self._possible_actions = np.arange(1)
            elif n_seen == self.n_item:
                self._possible_actions = np.arange(self.n_item)
            else:
                already_seen = np.arange(self.n_item)[seen]
                new = np.max(already_seen)+1
                self._possible_actions = list(already_seen) + [new]
        return self._possible_actions

        # return np.arange(N_ITEM)

    def take_action(self, action, fake=True, root=False):
        """Returns the state which results from taking action 'action'"""
        n_pres = self.n_pres.copy()
        delta = self.delta.copy()
        assert isinstance(action, int) or isinstance(action, np.int64), f"Action if of type {type(action)}"
        n_pres[action] += 1

        # Increment delta for all items
        delta[:] += 1
        # ...except the one for the selected design that equal one
        delta[action] = 1
        t = self.t+1

        if action in self.children:
            new_state = self.children[action]
        else:
            new_state = LearnerState(
                parent=self, delta=delta,
                n_pres=n_pres, horizon=self.horizon,
                learnt_thr=self.learnt_thr, param=self.param,
                root_action=self.root_action
            )

        # If not considered only prospectively
        if not fake:
            new_state.t = 0
            new_state.cumulative_reward = - new_state.get_instant_reward()
        else:
            new_state.t = t
            new_state.cumulative_reward = \
                self.cumulative_reward + new_state.get_instant_reward()

        if root is True:
            new_state.root_action = action

        return new_state

    def is_terminal(self):
        """Returns whether this state is a terminal state"""
        return self.t >= self.horizon

    def get_instant_reward(self):
        """"Returns the INSTANT reward for this state"""

        if self._instant_reward is not None:
            return self._instant_reward
        else:
            seen = self.n_pres[:] > 0

            fr = self.param[0] * (1 - self.param[1]) ** (self.n_pres[seen] - 1)

            p = np.exp(-fr * self.delta[seen])
            self._instant_reward = np.sum(p > self.learnt_thr)
            return self._instant_reward

    def get_reward(self):
        """"Returns the CUMULATIVE reward for this state"""
        return self.cumulative_reward / self.horizon
    # def __str__(self):
    #     return f"State: "
