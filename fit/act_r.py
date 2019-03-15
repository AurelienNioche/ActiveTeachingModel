import numpy as np
import scipy.optimize
from multiprocessing import cpu_count

from model.learner import ActRLearner
from task.parameters import n_possible_replies

from . generic import log_likelihood_sum


def _objective(*args):

    p_choices = get_p_choices(*args)

    if p_choices is None:
        return np.inf

    return - log_likelihood_sum(p_choices)


def get_p_choices(parameters, questions, replies, n_items, use_p_correct):

    if type(parameters) == np.ndarray:
        d, tau, s = parameters
        if np.any(np.isnan(parameters)):
            return None
    else:
        d, tau, s = parameters["d"], parameters["tau"], parameters["s"]

    learner = ActRLearner(n_items=n_items, t_max=len(questions), n_possible_replies=n_possible_replies, d=d, tau=tau, s=s)
    return learner.get_p_choices(questions=questions, replies=replies, use_p_correct=use_p_correct)


def fit(questions, replies, n_items, use_p_correct=False, bounds=((0, 1.0), (0.00, 5), (0.0000001, 10))):

    # res = scipy.optimize.minimize(
    #     _objective, np.array([0.5, 0.0001, 0.00001]), args=(questions, replies, n_items),
    #     bounds=((0.4, 0.6), (0.00001, 0.001), (0.005, 0.015)))  # method=SLSQP

    res = scipy.optimize.differential_evolution(
        _objective, args=(questions, replies, n_items, use_p_correct),
        bounds=bounds,
        updating='deferred', workers=cpu_count()
    )

    d, tau, s = res.x
    best_param = {"d": d, "tau": tau, "s": s}

    return best_param
