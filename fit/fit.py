import numpy as np
import scipy.optimize
from hyperopt import hp, fmin, tpe

from utils import utils

IDX_PARAMETERS = 0
IDX_MODEL = 1
IDX_ARGUMENTS = 2


class Fit:

    def __init__(self, tk, model, data=None,
                 verbose=False, method='de', **kwargs):

        self.kwargs = kwargs
        self.method = method

        self.tk = tk
        self.model = model
        self.data = data

        self.best_value = None
        self.best_param = None

        self.history_eval_param = []
        self.obj_values = []

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

    def objective(self, param, keep_in_history=True):

        agent = self.model(param=param, tk=self.tk)
        p_choices_ = agent.get_p_choices(data=self.data,
                                         **self.kwargs)

        if p_choices_ is None or np.any(np.isnan(p_choices_)):
            # print("WARNING! Objective function returning 'None'")
            value = np.inf

        else:
            value = - self._log_likelihood_sum(p_choices_)

        if keep_in_history:
            self.obj_values.append(value)
            self.history_eval_param.append(param)
        return value

    def evaluate(self, max_iter=1000, data=None, **kwargs):

        if data is not None:
            self.data = data

        if self.method == 'tpe':  # Tree of Parzen Estimators

            space = [hp.uniform(*b) for b in self.model.bounds]
            best_param = fmin(fn=self.objective, space=space,
                              algo=tpe.suggest,
                              max_evals=kwargs['max_evals'])

        else:

            bounds_scipy = [b[-2:] for b in self.model.bounds]

            if self.method == 'de':
                res = scipy.optimize.differential_evolution(
                    func=self.objective, bounds=bounds_scipy,
                    maxiter=kwargs['maxiter'], workers=kwargs['workers']
                )
            else:
                x0 = np.zeros(len(self.model.bounds)) * 0.5
                res = scipy.optimize.minimize(fun=self.objective,
                                              bounds=bounds_scipy, x0=x0)

            best_param = {b[0]: v for b, v in zip(self.model.bounds, res.x)}
            if self.verbose:
                print(f"{res.message} [best loss: {res.fun}]")
            if not res.success:
                if self.verbose:
                    print(
                        f"The fit did not succeed with method {self.method}.")
                return None

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
