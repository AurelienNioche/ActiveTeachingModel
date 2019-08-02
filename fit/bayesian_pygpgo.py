from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO

import numpy as np
from fit.fit import Fit

import sys


class BayesianPYGPGOFit(Fit):

    def __init__(self, tk, model, data, verbose=False, **kwargs):

        super().__init__(tk=tk, model=model, data=data, verbose=verbose,
                         method='BayesianGPyOpt',
                         **kwargs)
        # self.best_value = None
        self.best_param = None

        # self.optimizer = BayesianOptimization()

    def _objective(self, **param):

        agent = self.model(param=param, tk=self.tk)
        p_choices_ = agent.get_p_choices(data=self.data,
                                         **self.kwargs)
        assert None not in p_choices_
        return np.mean(p_choices_)

    def evaluate(self, **kwargs):

        param = {
            f'{b[0]}': ('cont', [b[1], b[2]])
            for b in self.model.bounds
        }

        f = self._objective

        sexp = squaredExponential()
        gp = GaussianProcess(sexp)
        acq = Acquisition(mode='ExpectedImprovement')

        np.random.seed(123)
        gpgo = GPGO(gp, acq, f, param)
        gpgo.run(**kwargs)
        r = gpgo.getResult()

        if self.best_param is None:
            self.best_param = {}
        self.best_param.update(r[0])

        return self.get_stats()
