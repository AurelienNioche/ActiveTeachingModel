import numpy as np
from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO


def f(x, y, z):
    return x +2*y -z


sexp = squaredExponential()
gp = GaussianProcess(sexp)
acq = Acquisition(mode='ExpectedImprovement')
param = {
    'x': ('cont', [0, 1]),
    'y': ('cont', [0, 1]),
    'z': ('cont', [0, 1])
}

np.random.seed(23)
gpgo = GPGO(gp, acq, f, param)
gpgo.run(max_iter=30)
print(gpgo.getResult())
