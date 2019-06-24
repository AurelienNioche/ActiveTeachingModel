import numpy as np

import scipy.optimize
from hyperopt import hp, fmin, tpe

from utils import utils

IDX_PARAMETERS = 0
IDX_MODEL = 1
IDX_ARGUMENTS = 2

MAX_EVAL = 500  # Only if tpe


class Fit:

    def __init__(self, tk, model, data, verbose=False, method='de', **kwargs):

        self.kwargs = kwargs
        self.method = method

        self.tk = tk
        self.model = model
        self.data = data

        self.verbose = verbose

    def _bic(self, lls, k):
        """
        :param lls: log-likelihood sum
        :param k: number of parameters
        :return: BIC
        """
        return -2 * lls + np.log(self.tk.t_max) * k

    @classmethod
    def _log_likelihood_sum(cls, p_choices):

        try:
            return np.sum(np.log(p_choices))
        except FloatingPointError:
            return - np.inf

    def _model_stats(self, p_choices, best_param):

        mean_p = np.mean(p_choices)
        lls = self._log_likelihood_sum(p_choices)
        bic = self._bic(lls=lls, k=len(best_param))

        return mean_p, lls, bic

    def evaluate(self):

        def objective(param):

            agent = self.model(param=param, tk=self.tk)
            p_choices_ = agent.get_p_choices(data=self.data,
                                             **self.kwargs)

            if p_choices_ is None or np.any(np.isnan(p_choices_)):
                # print("WARNING! Objective function returning 'None'")
                to_return = np.inf

            else:
                to_return = - self._log_likelihood_sum(p_choices_)

            return to_return

        if self.method == 'tpe':  # Tree of Parzen Estimators

            space = [hp.uniform(*b) for b in self.model.bounds]
            best_param = fmin(fn=objective, space=space, algo=tpe.suggest,
                              max_evals=MAX_EVAL)

        elif self.method in \
                ('de', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP'):

            bounds_scipy = [b[-2:] for b in self.model.bounds]

            if self.verbose:
                print("Finding best parameters...", end=' ')

            if self.method == 'de':
                res = scipy.optimize.differential_evolution(
                        func=objective, bounds=bounds_scipy)
            else:
                x0 = np.zeros(len(self.model.bounds)) * 0.5
                res = scipy.optimize.minimize(fun=objective,
                                              bounds=bounds_scipy, x0=x0)

            best_param = {b[0]: v for b, v in zip(self.model.bounds, res.x)}
            if self.verbose:
                print(f"{res.message} [best loss: {res.fun}]")
            if not res.success:
                raise Exception(f"The fit did not succeed with method "
                                f"{self.method}.")

        else:
            raise Exception(f"Method {self.method} is not defined")

        # Get probabilities with best param
        learner = self.model(param=best_param, tk=self.tk)
        p_choices = learner.get_p_choices(data=self.data,
                                          **self.kwargs)

        # Compute bic, etc.
        mean_p, lls, bic = self._model_stats(p_choices=p_choices,
                                             best_param=best_param)

        if self.verbose:
            self._print(self.model.__name__, best_param, mean_p, lls, bic)
        return \
            {
                "best_param": best_param,
                "mean_p": mean_p,
                "lls": lls,
                "bic": bic
            }

    def _print(self, model_name, best_param, mean_p, lls, bic):

        dsp_bp = ''.join(f'{k}={round(best_param[k], 3)}, '
                         for k in sorted(best_param.keys()))

        dsp_kwargs = f'[{utils.dic2string(self.kwargs)}]' if len(self.kwargs) \
            else ''

        print(f"[{model_name} - '{self.method}']{dsp_kwargs}"
              f"Best param: " + dsp_bp +
              f"LLS: {round(lls, 2)}, " +
              f'BIC: {round(bic, 2)}, mean(P): {round(mean_p, 3)}\n')

        print("\n" + ''.join("_" * 10) + "\n")
