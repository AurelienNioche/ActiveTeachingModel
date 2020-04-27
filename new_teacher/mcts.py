import numpy as np

from mcts.mcts import MCTS
from mcts.state import LearnerState

from . abstract import Teacher


class MCTSTeacher(Teacher):

    def __init__(self, n_item, reward,
                 n_iter_per_ss, n_iter_between_ss,
                 terminal_t,
                 # horizon=20,
                 iteration_limit=500):
        super().__init__(
            n_item=n_item,
            n_iter_per_ss=n_iter_per_ss,
            n_iter_between_ss=n_iter_between_ss)

        self.iteration_limit = iteration_limit

        self.learner_state = \
            LearnerState(
                n_pres=np.zeros(self.n_item, dtype=int),
                delta=np.zeros(self.n_item, dtype=int),
                # horizon=horizon,
                reward=reward,
                n_iter_per_ss=self.n_iter_per_ss,
                n_iter_between_ss=self.n_iter_between_ss,
                t=0,
                c_iter_session=0,
                terminal_t=terminal_t
            )

    def ask(self):

        # self.learner_state.reset()

        m = MCTS(iteration_limit=self.iteration_limit)
        best_action = m.run(initial_state=self.learner_state)
        self.learner_state = self.learner_state.take_action(best_action)

        return best_action
