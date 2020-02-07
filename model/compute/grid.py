from itertools import product

import numpy as np


def compute_grid_param(grid_size, bounds):
    return np.asarray(list(
        product(*[
            np.linspace(*b, grid_size)
            for b in bounds])))
