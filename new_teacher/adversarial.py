import numpy as np

from . bruteforce import BruteForceTeacher


class AdversarialTeacher(BruteForceTeacher):

    def __init__(self, param, n_item, learnt_threshold,
                 horizon=2, verbose=0):

        super().__init__(
            param=param, n_item=n_item,
            learnt_threshold=learnt_threshold,
            horizon=horizon, verbose=verbose
        )

    def ask(self):
        if self.verbose:
            print("*" * 40)
            print(f"T = {self.t}")
            print("*" * 40)
            print()

        initial_state = self.learner_state
        learner_state = initial_state
        actions = learner_state.get_possible_actions()

        if self.verbose:
            print()
            print("Possible actions " + "*" * 10)
            print(f"t={learner_state.t}, possible actions={actions}")
            print()
            print("Evaluation" + "*" * 10)

        min_values = np.zeros(len(actions))

        for i, a in enumerate(actions):
            learner_state = learner_state.take_action(a)
            _, values = self.explore_tree(initial_state=learner_state)

            min_values[i] = np.min(values)

        max_min_value = np.max(min_values)
        a_actions = np.asarray(actions)
        best_action = np.random.choice(a_actions[min_values[:] == max_min_value])
        self.learner_state = self.learner_state.take_action(best_action)

        self.t += 1
        return best_action
