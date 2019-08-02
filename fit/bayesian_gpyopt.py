import sys

from GPyOpt.methods import BayesianOptimization

import numpy as np
from fit.fit import Fit


class BayesianGPyOptFit(Fit):

    def __init__(self, tk, model, data, verbose=False, **kwargs):

        super().__init__(tk=tk, model=model, data=data, verbose=verbose,
                         method='BayesianGPyOpt',
                         **kwargs)

        # self.best_value = None
        self.best_param = None

        # self.optimizer = BayesianOptimization()

    def _objective(self, param):

        param = param[0]

        agent = self.model(param=param, tk=self.tk)
        p_choices_ = agent.get_p_choices(data=self.data,
                                         **self.kwargs)

        if p_choices_ is None or np.any(np.isnan(p_choices_)):

            to_return = np.inf

        else:
            to_return = - self._log_likelihood_sum(p_choices_)

        return to_return  # * 10**100

    def evaluate(self, **kwargs):

        domain = [
            {'name': f'{b[0]}',
             'type': 'continuous',
             'domain': (b[1], b[2])
             } for b in self.model.bounds
        ]

        myBopt = BayesianOptimization(f=self._objective,
                                      domain=domain)
        myBopt.run_optimization(max_iter=kwargs['max_iter'])

        best_param_list = myBopt.x_opt
        self.best_param = {b[0]: v for b, v in
                           zip(self.model.bounds,
                               best_param_list)}

        return self.get_stats()
