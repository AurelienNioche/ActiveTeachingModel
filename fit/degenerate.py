from abc import ABC

import numpy as np
from . abstract_class import Fit
from . objective import objective
from learner.generic import Learner


class PerfectStudent(Learner, ABC):

    bounds = ()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def p_recall(self, item, time_index=None, time=None):
        return 1


class Degenerate(Fit):

    def __init__(self, **kwargs):

        super().__init__(model=PerfectStudent, **kwargs)

    def _run(self, bounds, **kwargs):

        self.best_value = self.objective(param=None)
        return True
