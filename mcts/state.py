"""
Adapted from: https://github.com/pbsinclair42/MCTS/blob/master/mcts.py
"""
import numpy as np

# np.seterr(all='raise')


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
                 c_iter_ss,
                 t,
                 c_iter,
                 reward,
                 terminal_t,
                 n_iter_per_ss,
                 n_iter_between_ss,
                 horizon=None,
                 ref_point=None,
                 # terminal_t=None,
                 # action=None,
                 # parent=None
                 ):

        self.n_pres = n_pres
        self.delta = delta

        self.t = t
        self.c_iter = c_iter
        self.c_iter_ss = c_iter_ss

        self.n_iter_per_ss = n_iter_per_ss
        self.n_iter_between_ss = n_iter_between_ss

        self.terminal_t = terminal_t
        self.horizon = horizon
        self.ref_point = ref_point

        # if self.horizon is not None:
        #     self._is_terminal = self.rel_t >= self.horizon
        # if self.terminal_t is not None:
        #     self._is_terminal = self.t == self.terminal_t
        # else:
        #     raise ValueError("Either 'horizon' of 't_final' should be defined")

        self.reward = reward

        self.n_item = self.n_pres.size

        self._instant_reward = None
        self._mean_reward = None
        self._possible_actions = None

        # self.parent = parent
        self.children = dict()

        self._learner_param = None

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

    def take_action(self, action):
        """Returns the state which results from taking action 'action'"""

        if action in self.children:
            new_state = self.children[action]

        else:
            n_pres = self.n_pres.copy()
            delta = self.delta.copy()

            n_pres[action] += 1

            # Increment delta for all items
            delta[:] += 1
            # ...except the one for the selected design that equal one
            delta[action] = 1

            t = self.t + 1
            c_iter = self.c_iter + 1

            c_iter_session = self.c_iter_ss + 1
            if c_iter_session >= self.n_iter_per_ss:
                delta[:] += self.n_iter_between_ss
                t += self.n_iter_between_ss
                c_iter_session = 0

            new_state = LearnerState(
                delta=delta,
                n_pres=n_pres,
                horizon=self.horizon,
                ref_point=self.ref_point,
                c_iter_ss=c_iter_session,
                reward=self.reward,
                n_iter_between_ss=self.n_iter_between_ss,
                n_iter_per_ss=self.n_iter_per_ss,
                t=t,
                c_iter=c_iter,
                terminal_t=self.terminal_t
            )

            self.children[action] = new_state

        return new_state

    def is_terminal(self):
        """Returns whether this state is a terminal state"""
        # print("ref point", self.ref_point.c_iter)
        # print("c_iter", self.c_iter)
        # print("terminal t", self.terminal_t)
        if self.t > self.terminal_t:
            raise ValueError(f"{self.t} > {self.terminal_t}")
        elif self.t == self.terminal_t:
            return True
        elif self.horizon is not None \
                and (self.c_iter - self.ref_point.c_iter) >= self.horizon:
            # print("Horizon reached")
            return True
        else:
            return False

    def get_instant_reward(self):
        """"Returns the INSTANT reward for this state"""
        if self._instant_reward is None:
            self._instant_reward = self.reward.reward(
                n_pres=self.n_pres, delta=self.delta,
                t=self.t
            )
        return self._instant_reward

    # def get_mean_reward(self):
    #     """"Returns the MEAN reward up to this state"""
    #     if self._mean_reward is None:
    #         self._mean_reward = np.mean(
    #             self.hist_reward + [self.get_instant_reward(), ])
    #
    #     return self._mean_reward

    def get_reward(self):

        return self.get_instant_reward()

    def __str__(self):
        return f"State: {self.t}, possible actions: {self.get_possible_actions()}"
        # else:
        #     return self.get_mean_reward()
        # print(f'mean: {mean}')

    def reset(self):

        self._instant_reward = None
        self.children = {}



    # def reset(self):
    #
    #     self.rel_t = 0
    #     self.hist_reward = []
    #     self.hist_action = []
