import sys

from bayes_opt import BayesianOptimization

import numpy as np
from fit.fit import Fit


class BayesianFit(Fit):

    def __init__(self, tk, model, data, verbose=False, **kwargs):

        super().__init__(tk=tk, model=model, data=data, verbose=verbose,
                         method='Bayesian',
                         **kwargs)

    def _objective(self, **param):

        agent = self.model(param=param, tk=self.tk)
        p_choices_ = agent.get_p_choices(data=self.data,
                                         **self.kwargs)

        if p_choices_ is None or np.any(np.isnan(p_choices_)):
            to_return = 0

        else:
            to_return = np.mean(p_choices_)

        return to_return

    def evaluate(self, **kwargs):

        pbounds = {tup[0]: (tup[1], tup[2]) for tup in self.model.bounds}

        optimizer = BayesianOptimization(
            f=self._objective,
            pbounds=pbounds,
            random_state=1,
            verbose=kwargs['verbose']
        )

        if self.best_param is not None:
            optimizer.probe(
                params=self.best_param,
                lazy=True
            )

        optimizer.maximize(init_points=kwargs['init_points'],
                           n_iter=kwargs['n_iter'])

        self.best_param = optimizer.max['params']

        self.best_value = optimizer.max['target']

        return self.get_stats()
