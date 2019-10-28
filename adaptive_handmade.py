import numpy as np

from scipy.special import logsumexp
from itertools import product

from learner.half_life import HalfLife

from adaptive_design.run import run
from adaptive_design.engine.classic import AdaptiveClassic
from adaptive_design.engine.revised import AdaptiveRevised
from adaptive_design.learner.fake import FakeModel

EPS = np.finfo(np.float).eps


def main_revised():

    grid_size = 100
    n_design = 10

    possible_design = np.arange(n_design)
    learner_model = HalfLife
    true_param = {
        "beta": 0.02,
        "alpha": 0.2
    }

    # possible_design = np.linspace(0, 10, 100)
    # learner_model = FakeModel
    # true_param = {
    #     "b0": 2,
    #     "b1": 3,
    # }

    num_trial = 200

    run(adaptive_engine=AdaptiveRevised,
        learner_model=learner_model,
        true_param=true_param,
        possible_design=possible_design,
        grid_size=grid_size,
        num_trial=num_trial)


def main():

    np.random.seed(123)

    # possible_design = np.arange(10)
    # learner_model = HalfLife
    # true_param = {
    #     "beta": 0.02,
    #     "alpha": 0.2
    # }

    possible_design = np.linspace(0, 10, 100)
    learner_model = FakeModel
    true_param = {
        "b0": 2,
        "b1": 3,
    }

    grid_size = 100
    num_trial = 200

    run(adaptive_engine=AdaptiveClassic,
        learner_model=learner_model,
        true_param=true_param,
        possible_design=possible_design,
        grid_size=grid_size,
        num_trial=num_trial)


if __name__ == '__main__':
    main()
