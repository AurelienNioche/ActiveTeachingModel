import numpy as np

from mcts.state import LearnerState


class BruteForceTeacher:

    def __init__(self, param, n_item, learnt_threshold,
                 horizon=2, verbose=0):

        self.t = 0

        self.learner_state = \
            LearnerState(
                n_pres=np.zeros(n_item, dtype=int),
                delta=np.zeros(n_item, dtype=int),
                learnt_thr=learnt_threshold,
                horizon=horizon,
                param=param)

        self.verbose = verbose

    def ask(self):
        if self.verbose:
            print("*" * 40)
            print(f"T = {self.t}")
            print("*" * 40)
            print()

        actions, values = self.explore_tree(self.learner_state)
        best_action = self.pickup_best_action(actions=actions,
                                              values=values)

        self.learner_state = self.learner_state.take_action(best_action)

        self.t += 1
        return best_action

    def explore_tree(self, initial_state):

        done = False

        values = []
        root_actions = []

        learner_state = initial_state
        learner_state.reset()

        while True:
            actions = learner_state.get_possible_actions()

            if self.verbose == 2:
                print()
                print("Possible actions " + "*" * 10)
                print(f"t={learner_state.t}, possible actions={actions}")
                print()
                print("Evaluation" + "*"*10)

            for a in actions:

                if a not in learner_state.children:
                    if self.verbose == 2:
                        print(f"evaluate action {a} at t={learner_state.t}")
                    learner_state = learner_state.take_action(action=a)
                    if self.verbose == 2:
                        print(f"New learner state at t={learner_state.t} "
                              f"with r={learner_state.get_instant_reward()}")
                    break

            if learner_state.is_terminal():
                if self.verbose == 2:
                    print("New state is terminal.")
                values.append(learner_state.get_reward())
                root_actions.append(learner_state.action[0])

                fully_expanded = True
                while fully_expanded:
                    if self.verbose == 2:
                        print("Fully expanded or terminal. Taking parent.")
                    learner_state = learner_state.parent
                    actions = learner_state.get_possible_actions()
                    fully_expanded = len(learner_state.children) == len(
                        actions)

                    root = learner_state == initial_state
                    if fully_expanded and root:
                        if self.verbose == 2:
                            print("Tree is fully expanded.")
                        done = True
                        break

            if done:
                if self.verbose == 2:
                    print("Ready to provide action.")
                break

        return root_actions, values

    def pickup_best_action(self, actions, values):

        a_values = np.asarray(values)
        a_actions = np.asarray(actions)
        max_value = np.max(a_values)
        best_action = np.random.choice(a_actions[a_values == max_value])

        if self.verbose:
            print(f"Action-Value list={[(s,v) for s, v in zip(actions, values)]}")
            print("Selected action", best_action)
            print("Max value", max_value)
            print("*" * 40)
        return best_action
