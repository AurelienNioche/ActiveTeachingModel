import numpy as np
from . abstract_class import Fit
from . objective import objective
from bot_client.learning_model.generic import Learner


class PerfectStudent(Learner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def p_recall(self, item, time=None):
        return 1

    def learn(self, **kwargs):
        pass


class Degenerate(Fit):

    def __init__(self, **kwargs):

        super().__init__(model=PerfectStudent, **kwargs)

    def _run(self):

        self.best_value = self.objective(param=None)
        return True
