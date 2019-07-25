from bayes_opt import BayesianOptimization

import numpy as np
from fit.fit import Fit


class BayesianFit(Fit):

    def __init__(self, tk, model, data, verbose=False, **kwargs):

        super().__init__(tk=tk, model=model, data=data, verbose=verbose,
                         method='Bayesian',
                         **kwargs)

        self.best_value = None
        self.best_param = None

        # self.optimizer = BayesianOptimization()

    def _objective(self, **param):

        agent = self.model(param=param, tk=self.tk)
        p_choices_ = agent.get_p_choices(data=self.data,
                                         **self.kwargs)

        if p_choices_ is None or np.any(np.isnan(p_choices_)):
            # print("WARNING! Objective function returning 'None'")
            to_return = np.finfo(np.float64).min * 100

        else:
            to_return = self._log_likelihood_sum(p_choices_)

        return to_return

    def evaluate(self, init_points=20, n_iter=20, verbose=2):

        pbounds = {tup[0]: (tup[1], tup[2]) for tup in self.model.bounds}

        optimizer = BayesianOptimization(
            f=self._objective,
            pbounds=pbounds,
            random_state=1,
            verbose=verbose
        )

        if self.best_param is not None:
            optimizer.probe(
                params=self.best_param,
                lazy=True
            )
        # try:
        res = optimizer.maximize(init_points=init_points, n_iter=n_iter)
        # except (StopIteration, FloatingPointError, ValueError):
        #     return

        self.best_param = res.max['params']

        self.best_value = res.max['target']
        if self.verbose:
            print(f"Best value: {self.best_value}")

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