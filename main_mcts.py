import numpy as np

from mcts.mcts import MCTS

from mcts.state import State


N_ITEM = 100


class LearnerState(State):

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

    def is_terminal(self, ):
        """Returns whether this state is a terminal state"""
        return False

    def get_reward(self):
        """"Returns the reward for this state"""
        return np.sum(self.seen == 1)

    def __str__(self):
        return f"State: {self.seen}"


def main():

    leaner_state = LearnerState(seen=np.zeros(N_ITEM, dtype=bool))
    for t in range(10):
        m = MCTS(iteration_limit=1000, verbose=False)
        action = m.run(initial_state=leaner_state)

        leaner_state = leaner_state.take_action(action)
        print(f"t={t}, action={action}, new state = {leaner_state.seen}")


if __name__ == "__main__":
    main()
