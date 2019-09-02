from bayes_opt import BayesianOptimization

import numpy as np
from fit.fit import Fit


class BayesianFit(Fit):

    def __init__(self, tk, model, data, verbose=False, **kwargs):

        super().__init__(tk=tk, model=model, data=data, verbose=verbose,
                         method='Bayesian',
                         **kwargs)

    def objective(self, keep_in_history=True, **param):

        agent = self.model(param=param, tk=self.tk)
        p_choices_ = agent.get_p_choices(data=self.data,
                                         stop_if_zero=False,
                                         use_p_correct=True,
                                         **self.kwargs)

        value = np.sum(p_choices_)

        if keep_in_history:
            self.obj_values.append(value)
            self.history_eval_param.append(param)
        return value

    def evaluate(self, data=None, verbose=2, **kwargs):

        """
        :param verbose:
        :param kwargs:
            - init_points
            - n_iter
        :return:
        """

        if data is not None:
            self.data = data

        pbounds = {tup[0]: (tup[1], tup[2]) for tup in self.model.bounds}

        optimizer = BayesianOptimization(
            f=self.objective,
            pbounds=pbounds,
            random_state=1,
            verbose=verbose)

        if self.best_param is not None:
            optimizer.probe(
                params=self.best_param,
                lazy=True
            )

        optimizer.maximize(**kwargs)

        self.best_param = optimizer.max['params']

        self.best_value = optimizer.max['target']

        return self.get_stats()
