import numpy as np

from mcts.mcts import MCTS
from mcts.state import LearnerState

from . abstract import Teacher
from . psychologist import Psychologist


class DynamicFeatures:

    def __init__(self):
        self.t = 0
        self.c_iter = 0
        self.param = None


class MCTSTeacher(Teacher):

    def __init__(self, n_item, reward,
                 n_iter_per_ss, n_iter_between_ss,
                 terminal_t,
                 horizon=20,
                 iteration_limit=500):
        super().__init__(
            n_item=n_item,
            n_iter_per_ss=n_iter_per_ss,
            n_iter_between_ss=n_iter_between_ss)

        self.iteration_limit = iteration_limit

        self.reference_point = DynamicFeatures()

        self.learner_state = \
            LearnerState(
                n_pres=np.zeros(self.n_item, dtype=int),
                delta=np.zeros(self.n_item, dtype=int),
                horizon=horizon,
                ref_point=self.reference_point,
                reward=reward,
                n_iter_per_ss=self.n_iter_per_ss,
                n_iter_between_ss=self.n_iter_between_ss,
                t=0,
                c_iter=0,
                c_iter_ss=0,
                terminal_t=terminal_t)

        self.c_iter = 0

    def ask(self):

        self.reference_point.c_iter = self.c_iter
        m = MCTS(iteration_limit=self.iteration_limit)
        best_action = m.run(initial_state=self.learner_state)
        self.learner_state = self.learner_state.take_action(best_action)
        self.c_iter += 1
        return best_action


class MCTSPsychologist(MCTSTeacher):

    def __init__(self,
                 n_item,
                 n_ss,
                 n_iter_per_ss,
                 n_iter_between_ss,
                 reward,
                 terminal_t,
                 param,
                 horizon=20,
                 iteration_limit=500):

        self.reward = reward

        super().__init__(
            n_item=n_item,
            n_iter_per_ss=n_iter_per_ss,
            n_iter_between_ss=n_iter_between_ss,
            reward=reward,
            terminal_t=terminal_t,
            horizon=horizon,
            iteration_limit=iteration_limit)

        self.psychologist = Psychologist(
            n_item=n_item,
            n_iter_per_ss=n_iter_per_ss,
            n_iter_between_ss=n_iter_between_ss,
            n_ss=n_ss,
            true_param=param)

    def ask(self):

        self.reward.param = self.psychologist.pm

        self.reference_point.c_iter = self.c_iter

        m = MCTS(iteration_limit=self.iteration_limit)

        self.learner_state.reset()
        item = m.run(initial_state=self.learner_state)
        self.update(item)
        return item

    def update(self, item):
        self.learner_state = self.learner_state.take_action(item)
        self.c_iter += 1
        self.psychologist.update(item=item, response=None)
