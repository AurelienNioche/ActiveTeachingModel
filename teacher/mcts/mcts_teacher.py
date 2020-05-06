import numpy as np
from tqdm import tqdm
from copy import deepcopy

from . mcts.mcts import MCTS
from . mcts.state import LearnerState


class ReferencePoint:

    def __init__(self):
        self.t = 0
        self.c_iter = 0


class MCTSTeacher:

    def __init__(self,
                 learner,
                 reward,
                 terminal_t,
                 horizon=20,
                 iteration_limit=500):

        self.iteration_limit = iteration_limit

        self.reference_point = ReferencePoint()

        self.learner_state = \
            LearnerState(
                learner=learner,
                t=0,
                c_iter=0,
                horizon=horizon,
                ref_point=self.reference_point,
                reward=reward,
                terminal_t=terminal_t)

        self.c_iter = 0

    def ask(self):
        m = MCTS(iteration_limit=self.iteration_limit)

        self.reference_point.c_iter = self.c_iter

        self.learner_state.reset()
        best_action = m.run(initial_state=self.learner_state)

        self.learner_state = self.learner_state.take_action(best_action)
        return best_action

    def update(self, item):
        self.c_iter += 1

    def teach(self, n_iter, seed=0):

        np.random.seed(seed)
        h = np.zeros(n_iter, dtype=int)

        for t in tqdm(range(n_iter)):
            item = self.ask()
            self.update(item)
            h[t] = item
        return h


class MCTSPsychologist(MCTSTeacher):

    def __init__(self,
                 psychologist,
                 learner,
                 reward,
                 terminal_t,
                 horizon=20,
                 iteration_limit=500):
        self.reward = reward

        self.learner = learner

        super().__init__(
            learner=deepcopy(learner),
            reward=reward,
            terminal_t=terminal_t,
            horizon=horizon,
            iteration_limit=iteration_limit)

        self.psychologist = psychologist

    def ask(self):

        m = MCTS(iteration_limit=self.iteration_limit)

        self.learner_state.reset()
        self.learner_state.learner.param = self.psychologist.pm
        self.reference_point.c_iter = self.c_iter
        item = m.run(initial_state=self.learner_state)
        return item

    def update(self, item):
        self.learner_state = self.learner_state.take_action(item)
        response = self.learner.reply(item)
        self.psychologist.update(item=item, response=response)
        self.learner.update(item)
        self.c_iter += 1
