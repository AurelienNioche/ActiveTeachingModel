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

        if p_choices_ is None or np.any(np.isnan(p_choices_)):
            # print("WARNING! Objective function returning 'None'")
            # to_return = -np.inf  # 10**2
            to_return = sys.float_info.max
            # to_return = np.inf

        else:
            to_return = -self._log_likelihood_sum(p_choices_)

        return to_return  # * 10**100

    def evaluate(self, max_iter=1000):

        param = {
            f'{b[0]}': ('cont', [b[1], b[2]])
            for b in self.model.bounds
        }

        f = self._objective

        sexp = squaredExponential()
        gp = GaussianProcess(sexp)
        acq = Acquisition(mode='ExpectedImprovement')

        np.random.seed(23)
        gpgo = GPGO(gp, acq, f, param)
        gpgo.run(max_iter=max_iter)
        r = gpgo.getResult()

        self.best_param = {}
        self.best_param.update(r[0])

        # self.best_value = res.max['target']
        # if self.verbose:
        #     print(f"Best value: {self.best_value}")

        return {
            'best_param': self.best_param
        }

    def get_stats(self):

        # Get probabilities with best param
        learner = self.model(param=self.best_param, tk=self.tk)
        p_choices = learner.get_p_choices(data=self.data,
                                          **self.kwargs)

        # Compute bic, etc.
        mean_p, lls, bic = self._model_stats(p_choices=p_choices,
                                             best_param=self.best_param)

        if self.verbose:
            self._print(self.model.__name__, self.best_param, mean_p, lls, bic)
        return \
            {
                "best_param": self.best_param,
                "mean_p": mean_p,
                "lls": lls,
                "bic": bic
            }
