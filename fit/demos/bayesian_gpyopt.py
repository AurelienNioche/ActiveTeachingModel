# --- Load GPyOpt
from GPyOpt.methods import BayesianOptimization
import numpy as np


# --- Define your problem
def f(args):
    x, y, z = args[0]
    return x + 2*y - z


domain = [
    {
        'name': 'x',
        'type': 'continuous',
        'domain': (0, 1)
    },
    {
        'name': 'y',
        'type': 'continuous',
        'domain': (0, 1)
    },

    {
        'name': 'z',
        'type': 'continuous',
        'domain': (0, 1)
    }
]

# --- Solve your problem
myBopt = BayesianOptimization(f=f, domain=domain)
myBopt.run_optimization(max_iter=15)

print(myBopt.x_opt)

# myBopt.plot_acquisition()

myBopt.plot_convergence()