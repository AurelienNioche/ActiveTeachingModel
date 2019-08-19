from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO, EventLogger

import multiprocessing as mp

import numpy as np
from fit.pygpgo.objective import objective


import time


class MyGPGO(GPGO):

    # noinspection PyMissingConstructor
    def __init__(self, surrogate, acquisition, f, parameter_dict, n_jobs=1,
                 verbose=True):

        # super().__init__(surrogate, acquisition, f, parameter_dict, n_jobs)

        self.GP = surrogate
        self.A = acquisition
        self.f = f
        self.parameters = parameter_dict
        self.n_jobs = n_jobs

        self.parameter_key = list(parameter_dict.keys())
        self.parameter_value = list(parameter_dict.values())
        self.parameter_type = [p[0] for p in self.parameter_value]
        self.parameter_range = [p[1] for p in self.parameter_value]

        self.history = []

        self.verbose = verbose
        if verbose:
            self.logger = EventLogger(self)

    def run(self, max_iter=10, init_evals=3, timeout=None,
            resume=False, init_param=None):

        start = time.time()

        if not resume:
            self.init_evals = init_evals
            # Add init_param argument
            self._firstRun(n_eval=self.init_evals, init_param=init_param)
            if self.verbose:
                self.logger._printInit(self)
        for iteration in range(max_iter):

            if timeout is not None and time.time() - start > timeout:
                break

            self._optimizeAcq()
            self.updateGP()
            if self.verbose:
                self.logger._printCurrent(self)

    def _firstRun(self, n_eval=3, init_param=None):

        # Add this block
        if init_param is not None:
            self.X = np.empty((n_eval+1, len(self.parameter_key)))
            self.y = np.empty((n_eval+1,))
            s_param = init_param
            s_param_val = [s_param[k] for k in self.parameter_key]
            self.X[0] = s_param_val
            self.y[0] = self.f(**s_param)

            iterator = range(1, n_eval+1)

        else:
            self.X = np.empty((n_eval, len(self.parameter_key)))
            self.y = np.empty((n_eval,))
            iterator = range(0, n_eval)

        for i in iterator:
            s_param = self._sampleParam()
            s_param_val = list(s_param.values())
            self.X[i] = s_param_val
            self.y[i] = self.f(**s_param)

        self.GP.fit(self.X, self.y)
        self.tau = np.max(self.y)
        self.history.append(self.tau)

    def updateGP(self):
        """
        Updates the internal model with the next acquired point and its evaluation.
        """
        kw = {param: self.best[i] for i, param in enumerate(self.parameter_key)}
        f_new = self.f(**kw)
        self.GP.update(np.atleast_2d(self.best), np.atleast_1d(f_new))
        self.tau = np.max(self.GP.y)
        self.history.append(self.tau)


class PYGPGOFit:

    def __init__(self, verbose=False,
                 init_evals=3,
                 max_iter=10,
                 timeout=None,
                 n_jobs=mp.cpu_count()):

        self.best_value = None
        self.best_param = None

        self.eval_param = []

        self.hist_best_param = []
        self.hist_best_value = []

        self.n_jobs = n_jobs
        self.init_evals = init_evals
        self.max_iter = max_iter
        self.timeout = timeout

        self.verbose = verbose

        self.model = None
        self.tk = None
        self.data = None

    def objective(self, **param):

        self.eval_param.append(param)
        return objective(model=self.model, data=self.data, tk=self.tk,
                         param=param)

    def evaluate(self, model, item):

        self.model = model
        self.

        self.eval_param = []

        sexp = squaredExponential()
        gp = GaussianProcess(sexp)
        acq = Acquisition(mode='ExpectedImprovement')
        f = self.objective
        param = {
            f'{b[0]}': ('cont', [b[1], b[2]])
            for b in self.model.bounds
        }

        # surrogate = BoostedTrees()

        opt = MyGPGO(gp, acq, f, param,
                     n_jobs=self.n_jobs,
                     verbose=self.verbose)

        opt.run(init_param=self.best_param,
                init_evals=self.init_evals,
                max_iter=self.max_iter,
                timeout=self.timeout)

        r = opt.getResult()

        self.best_param = dict(r[0])
        self.best_value = r[1]

        self.hist_best_param.append(self.best_param)
        self.hist_best_value.append(self.best_value)

        return self.best_param

    def stop(self):
        raise NotImplementedError
