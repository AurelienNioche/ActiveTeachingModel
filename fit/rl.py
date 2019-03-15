import numpy as np
import scipy.optimize

from model.learner import QLearner

from . generic import log_likelihood_sum


def _objective(*args):

    p_choices = get_p_choices(*args)

    if p_choices is None:
        return np.inf

    return - log_likelihood_sum(p_choices)


def get_p_choices(parameters, questions, replies, possible_replies, n_items, use_p_correct):

    if type(parameters) == np.ndarray:
        alpha, tau = parameters
        if np.any(np.isnan(parameters)):
            return None

    else:
        alpha, tau = parameters["alpha"], parameters["tau"]

    learner = QLearner(n_items=n_items, alpha=alpha, tau=tau)
    return learner.get_p_choices(questions=questions, replies=replies, possible_replies=possible_replies,
                                 use_p_correct=use_p_correct)


def fit(questions, replies, possible_replies, n_items, use_p_correct):

    res = scipy.optimize.differential_evolution(
        _objective, args=(questions, replies, possible_replies, n_items, use_p_correct),
        bounds=[(0.00, 1.00), (0.0015, 0.5)])

    # res = scipy.optimize.minimize(
    #     _objective, np.array([0.1, 0.01]), args=(questions, replies, possible_replies, n_items),
    #     bounds=((0.00, 1.00), (0.001, 0.1)))  # method=SLSQP

    alpha, tau = res.x
    best_param = {'alpha': alpha, 'tau': tau}

    return best_param
