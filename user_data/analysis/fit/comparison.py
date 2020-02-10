import numpy as np


def bic(lls, k, n):
    """
    :param lls: log-likelihood sum
    :param k: number of parameters
    :param n: number of iterations
    :return: BIC
    """
    return -2 * lls + np.log(n) * k
