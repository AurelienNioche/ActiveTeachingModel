"""
Adapted from: https://github.com/pbsinclair42/MCTS/blob/master/mcts.py
"""
# import numpy as np
from abc import abstractmethod

from copy import deepcopy

# np.seterr(all='raise')


class State:

    @abstractmethod
    def get_possible_actions(self):
        """Returns an iterable of all actions which can be taken
        from this state"""

    @abstractmethod
    def take_action(self, action):
        """Returns the state which results from taking action 'action'"""

    @abstractmethod
    def is_terminal(self, ):
        """Returns whether this state is a terminal state"""

    @abstractmethod
    def get_reward(self):
        """"Returns the reward for this state"""

    @abstractmethod
    def get_rollout_action(self):
        """"Returns one rollout action for this state"""


class LearnerState(State):

    def __init__(self,
                 learner,
                 reward,
                 rollout,
                 # terminal_t,
                 horizon=None,
                 dyn_param=None,
                 # action=None,
                 # parent=None
                 ):

        self.learner = learner

        # self.terminal_t = terminal_t
        self.horizon = horizon
        self.dyn_param = dyn_param

        # if self.horizon is not None:
        #     self._is_terminal = self.rel_t >= self.horizon
        # if self.terminal_t is not None:
        #     self._is_terminal = self.t == self.terminal_t
        # else:
        #     raise ValueError("Either 'horizon' of 't_final' should be defined")

        self.reward = reward
        self.rollout = rollout

        self.n_item = self.learner.n_item

        self._instant_reward = None
        self._possible_actions = None
        self._is_terminal = None
        self._learner_p_seen = None
        # self._mean_reward = None

        # self.parent = parent
        self.children = dict()

        # if parent is not None:
        #     self.hist_reward = \
        #         self.parent.hist_reward + [self.parent.get_instant_reward(), ]
        #     self.hist_action = \
        #         self.parent.hist_action + [action]
        #     self.rel_t = self.parent.rel_t + 1
        # else:
        #     self.t = 0
        #     self.hist_reward = []
        #     self.hist_action = []

    def get_possible_actions(self):
        """Returns an iterable of all actions which can be taken
        from this state"""
        if self._possible_actions is None:
            self._possible_actions = \
                self.rollout.get_possible_actions(
                    learner_seen=self.learner.seen)
        return self._possible_actions

    def take_action(self, action):
        """Returns the state which results from taking action 'action'"""

        if action in self.children:
            new_state = self.children[action]

        else:
            new_learner = deepcopy(self.learner)
            new_learner.update(item=action)
            new_state = LearnerState(
                horizon=self.horizon,
                dyn_param=self.dyn_param,
                reward=self.reward,
                learner=new_learner,
                rollout=self.rollout,
                # terminal_t=self.terminal_t
            )

            self.children[action] = new_state

        return new_state

    def is_terminal(self):
        """Returns whether this state is a terminal state"""
        if self._is_terminal is None:
            if self.learner.t > self.dyn_param.terminal_t:
                msg = f"learner T > terminal T\n" \
                      f"({self.learner.t} > {self.dyn_param.terminal_t})"
                raise ValueError(msg)
            elif self.learner.t == self.dyn_param.terminal_t:
                return True
            elif self.horizon is not None:
                if self.learner.c_iter - self.dyn_param.c_iter >= self.horizon:
                    return True
                else:
                    return False
            else:
                return False
        return self._is_terminal

    def get_reward(self):
        if self._instant_reward is None:
            self._instant_reward = \
                self.reward.reward(learner_p_seen=self.get_learner_p_seen())
        return self._instant_reward

    def get_rollout_action(self):
        # return np.random.choice(self.get_possible_actions())
        return self.rollout.get_action(learner_seen=self.learner.seen,
                                       learner_p_seen=self.get_learner_p_seen())

    def get_learner_p_seen(self):

        if self._learner_p_seen is None:
            self._learner_p_seen = self.learner.p_seen()
        return self._learner_p_seen

    def reset(self):
        self._instant_reward = None
        self._is_terminal = None
        self.children = {}

    # def get_instant_reward(self):
    #     """"Returns the INSTANT reward for this state"""
    #     if self._instant_reward is None:
    #         self._instant_reward = self.reward.reward(learner=self.learner)
    #     return self._instant_reward
    #
    # def get_reward(self):
    #     return self.get_instant_reward()

    # def get_mean_reward(self):
    #     """"Returns the MEAN reward up to this state"""
    #     if self._mean_reward is None:
    #         self._mean_reward = np.mean(
    #             self.hist_reward + [self.get_instant_reward(), ])
    #
    #     return self._mean_reward

