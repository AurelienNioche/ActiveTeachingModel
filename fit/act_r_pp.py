import numpy as np
import scipy.optimize

from model.learner import ActRPlusPlusLearner
from task.parameters import n_possible_replies

from . generic import log_likelihood_sum

from multiprocessing import cpu_count


def _objective(*args):

    p_choices = get_p_choices(*args)

    if p_choices is None:
        return np.inf

    return - log_likelihood_sum(p_choices)


def get_p_choices(parameters, questions, replies, n_items, c_graphic, c_semantic, use_p_correct):

    if type(parameters) == np.ndarray:
        d, tau, s, g, m, g_mu, g_sigma, m_mu, m_sigma = parameters
        if np.any(np.isnan(parameters)):
            return None
    else:
        d, tau, s, g, m, g_mu, g_sigma, m_mu, m_sigma = \
            parameters["d"], parameters["tau"], parameters["s"], parameters["g"], parameters["m"], \
            parameters["g_mu"], parameters["g_sigma"], parameters["m_mu"], parameters["m_sigma"]

    learner = ActRPlusPlusLearner(
        n_items=n_items, t_max=len(questions), c_graphic=c_graphic, c_semantic=c_semantic,
        n_possible_replies=n_possible_replies, d=d, tau=tau, s=s, g=g, m=m, g_mu=g_mu, g_sigma=g_sigma,
        m_mu=m_mu, m_sigma=m_sigma)
    return learner.get_p_choices(questions=questions, replies=replies, use_p_correct=use_p_correct)


def fit(questions, replies, n_items, c_graphic, c_semantic, use_p_correct=False,
        bounds=(
            (0, 1.0),  # d
            (0.00, 10),  # tau
            (0.0000001, 10),  # s
            (-1, 1),  # g
            (-1, 1),  # m
            (0, 1),  # g_mu
            (0.01, 5),  # g_sigma
            (0, 1),  # s_mu
            (0.01, 5))   # s_sigma
        ):

    # res = scipy.optimize.minimize(
    #     _objective, np.array([0.5, 0.000, 0.00001, 0.00, 0.00]),
    #     args=(questions, replies, n_items, c_graphic, c_semantic), bounds=bounds)
    # method=SLSQP

    res = scipy.optimize.differential_evolution(
        _objective,
        args=(questions, replies, n_items, c_graphic, c_semantic, use_p_correct), bounds=bounds,
        updating='deferred', workers=cpu_count())

    d, tau, s, g, m, g_mu, g_sigma, m_mu, m_sigma = res.x
    best_param = {"d": d, "tau": tau, "s": s, "g": g, "m": m, "g_mu": g_mu, "g_sigma": g_sigma, "m_mu": m_mu,
                  "m_sigma": m_sigma}

    return best_param
