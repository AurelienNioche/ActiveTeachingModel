import numpy as np

from mcts.mcts import MCTS
from mcts.state import LearnerState


class MCTSTeacher:

    def __init__(self,
                 n_item, learnt_threshold, param,
                 horizon=20, iteration_limit=500, verbose=0):

        self.t = 0
        self.iteration_limit = iteration_limit

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

        self.learner_state.reset()

        m = MCTS(iteration_limit=self.iteration_limit, verbose=self.verbose)
        best_action = m.run(initial_state=self.learner_state)
        self.learner_state = self.learner_state.take_action(best_action)

        if self.verbose:
            print("Selected action", best_action)
            print("*" * 40)

        self.t += 1
        return best_action