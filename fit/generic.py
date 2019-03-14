import numpy as np


def bic(lls, n, k):

    """
    :param lls: log-likelihood sum
    :param n: number of observations
    :param k: number of parameters
    :return: BIC
    """
    return -2 * lls + np.log(n) * k


def log_likelihood_sum(p_choices):

    return np.sum(np.log(p_choices))
