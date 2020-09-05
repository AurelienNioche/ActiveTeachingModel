import numpy as np


def generate_param(bounds, is_item_specific, n_item, methods=None):

    n_param = len(bounds)

    if methods is None:
        methods = np.array([np.random.uniform for _ in range(n_param)])

    if is_item_specific:
        param = np.zeros((n_item, n_param))
        for i, b in enumerate(bounds):
            param[:, i] = methods[i](b[0], b[1], n_item)
            # mean = np.random.uniform(b[0], b[1], size=n_item)
            # std = (b[1] - b[0]) * 0.1
            # v = np.random.normal(loc=mean, scale=std, size=n_item)
            # v[v < b[0]] = b[0]
            # v[v > b[1]] = b[1]
            # param[:, i] = v

    else:
        param = np.zeros(len(bounds))
        for i, b in enumerate(bounds):
            param[i] = methods[i](b[0], b[1])

    return param
