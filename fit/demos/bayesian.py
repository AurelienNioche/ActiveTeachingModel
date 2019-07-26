def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1

import numpy as np
def f(x, y, z):
    # x, y, z = args[0]
    return (6*x-2)**2*np.sin(12*x-4) +2*y -z


from bayes_opt import BayesianOptimization

# Bounded region of parameter space
# pbounds = {'x': (2, 4), 'y': (-3, 3)}

pbounds = {'x': (0,1), 'y': (0,1), 'z': (0,1)}

optimizer = BayesianOptimization(
    f=f,
    # f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=0,
    n_iter=30,
)

print(optimizer.max)
