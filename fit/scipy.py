import numpy as np
import scipy.optimize

from . abstract_class import Fit


class DifferentialEvolution(Fit):

    def __init__(self, **kwargs):

        super().__init__(method='de', **kwargs)

    def _run(self, bounds, **kwargs):

        """
        kwargs:  maxiter, workers
        """

        bounds_scipy = [b[-2:] for b in bounds]
        res = scipy.optimize.differential_evolution(
            func=self.objective,
            bounds=bounds_scipy,
            **kwargs)
        best_param_list = res.x
        self.best_value = res.fun

        success = res.success
        if not success:
            return res.success
        else:
            return best_param_list


class Minimize(Fit):

    def __init__(self, **kwargs):
        super().__init__(method='scipy', **kwargs)

    def _run(self, bounds, **kwargs):

        bounds_scipy = [b[-2:] for b in bounds]

        # A priori estimations
        x0 = np.zeros(len(bounds_scipy))
        for i, (min_, max_) in enumerate(bounds_scipy):
            x0[i] = (max_ - min_) / 2

        res = scipy.optimize.minimize(
            fun=self.objective,
            bounds=bounds_scipy,
            x0=x0,
            **kwargs
        )
        self.best_param = {b[0]: v for b, v in zip(self.model.bounds, res.x)}
        self.best_value = res.fun
        return res.success
