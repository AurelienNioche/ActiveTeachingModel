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

        agent = self.model(param=param, tk=self.tk)
        p_choices_ = agent.get_p_choices(data=self.data,
                                         **self.kwargs)

        if p_choices_ is None or np.any(np.isnan(p_choices_)):
            # print("WARNING! Objective function returning 'None'")
            to_return = 10**2
            #to_return = np.inf

        else:
            to_return = - self._log_likelihood_sum(p_choices_)

        return to_return  # * 10**100

    def evaluate(self, max_iter=1000):

        domain = [
            {'name': f'{b[0]}',
             'type': 'continuous',
             'domain': (b[1], b[2])
             } for b in self.model.bounds
        ]

        myBopt = BayesianOptimization(f=self._objective,
                                      domain=domain)
        myBopt.run_optimization(max_iter=15)

        best_param_list = myBopt.x_opt
        self.best_param = {b[0]: v for b, v in
                           zip(self.model.bounds,
                               best_param_list)}

        # self.best_value = res.max['target']
        # if self.verbose:
        #     print(f"Best value: {self.best_value}")

        return self.best_param

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