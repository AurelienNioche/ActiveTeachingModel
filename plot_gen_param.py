import matplotlib.pyplot as plt
import numpy as np


def generate_param(bounds, is_item_specific, n_item):

    if is_item_specific:
        print(f"Warning: methods for generating parameters "
              f"will be ignored as it is item specific")

        param = np.zeros((n_item, len(bounds)))
        for i, b in enumerate(bounds):
            # param[:, i] = methods[i](b[0], b[1], n_item)

            mean = np.random.uniform(b[0], b[1], size=n_item)
            std = (b[1] - b[0]) * 0.05

            v = np.random.normal(loc=mean, scale=std, size=n_item)

            v[v < b[0]] = b[0]
            v[v > b[1]] = b[1]
            param[:, i] = v

    else:
        param = np.zeros(len(bounds))
        for i, b in enumerate(bounds):
            param[i] = np.random.uniform(b[0], b[1])

    param = param.tolist()
    return param

def main():

    n_agent = 100

    gen_bounds = [[0.00000273, 0.00005], [0.42106842, 0.9999]]

    np.random.seed(123)
    param = np.zeros((n_agent, 2))
    for i in range(n_agent):
        param[i] = generate_param(bounds=gen_bounds,
                                  is_item_specific=False,
                                  n_item=500)




