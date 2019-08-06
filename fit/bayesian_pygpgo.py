from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.surrogates.BoostedTrees import BoostedTrees
from pyGPGO.GPGO import GPGO

import multiprocessing as mp

import numpy as np
from fit.fit import Fit

import sys


class MyGPGO(GPGO):

    def __init__(self, surrogate, acquisition, f, parameter_dict, n_jobs=1):

        super().__init__(surrogate, acquisition, f, parameter_dict, n_jobs)

    def run(self, max_iter=10, init_evals=3, resume=False, init_param=None,
            verbose=True):
        """
        Runs the Bayesian Optimization procedure.

        Parameters
        ----------
        max_iter: int
            Number of iterations to run. Default is 10.
        init_evals: int
            Initial function evaluations before fitting a GP. Default is 3.
        resume: bool
            Whether to resume the optimization procedure from the last evaluation. Default is `False`.

        """
        if not resume:
            self.init_evals = init_evals
            self._firstRun(n_eval=self.init_evals, init_param=init_param)
            if verbose:
                self.logger._printInit(self)
        for iteration in range(max_iter):
            self._optimizeAcq()
            self.updateGP()
            if verbose:
                self.logger._printCurrent(self)

    def _firstRun(self, n_eval=3, init_param=None):
        """
        Performs initial evaluations before fitting GP.

        Parameters
        ----------
        n_eval: int
            Number of initial evaluations to perform. Default is 3.

        """
        self.X = np.empty((n_eval, len(self.parameter_key)))
        self.y = np.empty((n_eval,))

        if init_param is not None:
            s_param = init_param
            s_param_val = [s_param[k] for k in self.parameter_key]
            self.X[0] = s_param_val
            self.y[0] = self.f(**s_param)

            iterator = range(1, n_eval)

        else:
            iterator = range(0, n_eval)

        for i in iterator:
            s_param = self._sampleParam()
            s_param_val = list(s_param.values())
            self.X[i] = s_param_val
            self.y[i] = self.f(**s_param)

        self.GP.fit(self.X, self.y)
        self.tau = np.max(self.y)
        self.history.append(self.tau)


class BayesianPYGPGOFit(Fit):

    def __init__(self, tk, model, data, verbose=False, seed=123,
                 n_jobs=mp.cpu_count() - 1,
                 **kwargs):

        super().__init__(tk=tk, model=model, data=data, verbose=verbose,
                         method='BayesianGPyOpt',
                         **kwargs)
        # self.best_value = None
        self.best_param = None

        self.history = []
        self.obj_values = []

        self.n_jobs = n_jobs

        np.random.seed(seed)

        # self.optimizer = BayesianOptimization()

    def objective(self, keep_in_history=True, **param):

        agent = self.model(param=param, tk=self.tk)
        p_choices_ = agent.get_p_choices(data=self.data,
                                         stop_if_zero=False,
                                         use_p_correct=True,
                                         **self.kwargs)

        value = np.sum(p_choices_)

        # assert None not in p_choices_
        if keep_in_history:
            self.obj_values.append(value)
            self.history.append(param)
        return value

    def evaluate(self, data=None, **kwargs):

        if data is not None:
            self.data = data

        param = {
            f'{b[0]}': ('cont', [b[1], b[2]])
            for b in self.model.bounds
        }

        f = self.objective

        sexp = squaredExponential()
        gp = GaussianProcess(sexp)

        # surrogate = BoostedTrees()
        acq = Acquisition(mode='ExpectedImprovement')

        opt = MyGPGO(gp, acq, f, param,
                     n_jobs=self.n_jobs)
        if self.best_param is None:
            print("No previous best param for now")
            self.best_param = {}
            opt.run(**kwargs)

        else:
            print(f"Best param is {self.best_param}")
            opt.run(init_param=self.best_param, **kwargs)

        r = opt.getResult()

        self.best_param.update(r[0])

        # return self.get_stats()
