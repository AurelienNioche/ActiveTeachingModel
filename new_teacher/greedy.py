import numpy as np

from mcts.mcts import MCTS
from mcts.state import LearnerState

from . abstract import Teacher


class GreedyTeacher(Teacher):

    def __init__(self, n_item, n_iter_per_ss, n_iter_between_ss,
                 reward,):
        super().__init__(
            n_item=n_item,
            n_iter_per_ss=n_iter_per_ss,
            n_iter_between_ss=n_iter_between_ss)

        self.learner_state = \
            LearnerState(
                n_pres=np.zeros(self.n_item, dtype=int),
                delta=np.zeros(self.n_item, dtype=int),
                reward=reward,
                n_iter_per_ss=self.n_iter_per_ss,
                n_iter_between_ss=self.n_iter_between_ss,
                t=0,
                c_iter_session=0,
            )

    def ask(self):

        self.learner_state.reset()

        actions = self.learner_state.get_possible_actions()
        r = np.zeros(len(actions))
        for i, a in enumerate(actions):
            state = self.learner_state.take_action(a)
            r[i] = state.get_instant_reward()

        best_action = actions[np.argmax(r)]
        self.learner_state = self.learner_state.take_action(best_action)
        return best_action
