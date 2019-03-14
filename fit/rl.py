import numpy as np
import scipy.optimize

from model.learner import QLearner

from . generic import bic, log_likelihood_sum


def _objective(parameters, questions, replies, possible_replies, n_items):

    p_choices = get_p_choices(parameters=parameters, questions=questions,
                              replies=replies, possible_replies=possible_replies, n_items=n_items)

    if p_choices is None:
        return np.inf

    return - log_likelihood_sum(p_choices)


def get_p_choices(parameters, questions, replies, possible_replies, n_items):

    if type(parameters) == np.ndarray:
        alpha, tau = parameters
        if np.any(np.isnan(parameters)):
            return None

    else:
        alpha, tau = parameters["alpha"], parameters["tau"]

    learner = QLearner(n_items=n_items, alpha=alpha, tau=tau)
    return learner.get_p_choices(questions=questions, replies=replies, possible_replies=possible_replies)


def fit(questions, replies, possible_replies, n_items):

    res = scipy.optimize.differential_evolution(
        _objective, args=(questions, replies, possible_replies, n_items),
        bounds=[(0.00, 1.00), (0.0015, 0.5)])

    # res = scipy.optimize.minimize(
    #     _objective, np.array([0.1, 0.01]), args=(questions, replies, possible_replies, n_items),
    #     bounds=((0.00, 1.00), (0.001, 0.1)))  # method=SLSQP

    alpha, tau = res.x
    lls = - res.fun

    best_param = {'alpha': alpha, 'tau': tau}
    n = len(questions)
    k = len(best_param)
    bic_v = bic(lls=lls, n=n, k=k)

    return lls, best_param, bic_v
