import numpy as np


def softmax(x, temp):
    try:
        return np.exp(x / temp) / np.sum(np.exp(x / temp))
    except (Warning, FloatingPointError) as w:
        print(x, temp)
        raise Exception(f'{w} [x={x}, temp={temp}]')


def temporal_difference(v, obs, alpha):

    return v + alpha*(obs-v)


def bic(lls, n, k):

    """
    :param lls: log-likelihood sum
    :param n: number of observations
    :param k: number of parameters
    :return: BIC
    """
    return -2 * lls + np.log(n) * k
