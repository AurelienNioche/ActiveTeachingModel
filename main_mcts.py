import numpy as np

from mcts.mcts import MCTS

from mcts.state import State


N_ITEM = 10
PARAM = (0.02, 0.2)
THR = 0.9


class BasicLearnerState(State):

    def __init__(self, seen):

        self.seen = seen

    def get_possible_actions(self):
        """Returns an iterable of all actions which can be taken
        from this state"""
        return np.arange(N_ITEM)

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

    def __init__(self, n_pres, delta):

        self.n_pres = n_pres
        self.delta = delta

        self._reward = None

    def get_possible_actions(self):
        """Returns an iterable of all actions which can be taken
        from this state"""
        return np.arange(N_ITEM)

    def take_action(self, action):
        """Returns the state which results from taking action 'action'"""
        n_pres = self.n_pres.copy()
        delta = self.delta.copy()

        n_pres[action] += 1

        # Increment delta for all items
        delta[:] += 1
        # ...except the one for the selected design that equal one
        delta[action] = 1
        return self.__class__(n_pres=n_pres, delta=delta)

    def is_terminal(self):
        """Returns whether this state is a terminal state"""
        return False

    def get_reward(self):
        """"Returns the reward for this state"""

        if self._reward is not None:
            return self._reward
        else:
            seen = self.n_pres[:] > 0

            fr = PARAM[0] * (1 - PARAM[1]) ** (self.n_pres[seen] - 1)

            p = np.exp(-fr * self.delta[seen])
            return np.sum(p > THR)

    # def __str__(self):
    #     return f"State: "


def main():

    leaner_state = LearnerState(n_pres=np.zeros(N_ITEM, dtype=bool),
                                delta=np.zeros(N_ITEM, dtype=int))
    for t in range(10):
        m = MCTS(iteration_limit=1000, verbose=False)
        action = m.run(initial_state=leaner_state)

        leaner_state = leaner_state.take_action(action)
        print(f"t={t}, action={action}")


if __name__ == "__main__":
    main()
