# --- Load GPyOpt
from GPyOpt.methods import BayesianOptimization
import numpy as np


# --- Define your problem
def f(args):
    x, y, z = args[0]
    return (6*x-2)**2*np.sin(12*x-4) +2*y -z


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
myBopt.run_optimization(max_iter=30)

print (myBopt.x_opt)

# myBopt.plot_acquisition()

myBopt.plot_convergence()