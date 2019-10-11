import numpy as np

from . objective import objective


class Fit:

    def __init__(self, model, verbose=False, method='de', **kwargs):

        self.kwargs = kwargs
        self.method = method

        self.model = model

        self.best_value = None
        self.best_param = None

        self.history_eval_param = []
        self.obj_values = []

        self.verbose = verbose

        self.hist_success = None
        self.hist_question = None
        self.task_param = None

    @staticmethod
    def _bic(lls, k, n):
        """
        :param lls: log-likelihood sum
        :param k: number of parameters
        :param n: number of iterations
        :return: BIC
        """
        return -2 * lls + np.log(n) * k

    @staticmethod
    def _log_likelihood_sum(p_choices):

        try:
            return np.sum(np.log(p_choices))
        except FloatingPointError:
            return - np.inf

    @classmethod
    def _model_stats(cls, p_choices, best_param):

        mean_p = np.mean(p_choices)
        lls = cls._log_likelihood_sum(p_choices)
        bic = cls._bic(lls=lls, k=len(best_param), n=len(p_choices))

        return {"mean_p": mean_p, "lls": lls, "bic": bic}

    def objective(self, param, keep_in_history=True):

        value = objective(
            model=self.model,
            hist_success=self.hist_success,
            hist_question=self.hist_question,
            task_param=self.task_param,
            param=param
        )

        if keep_in_history:
            self.obj_values.append(value)
            self.history_eval_param.append(param)
        return value

    def evaluate(self, hist_question, hist_success, task_param, **kwargs):

        self.hist_question = hist_question
        self.hist_success = hist_success
        self.task_param = task_param

        if 'unknown_param' in task_param:
            param_names = sorted(task_param['unknown_param'].keys())

        else:
            param_names = sorted(self.model.bounds.keys())

        bounds = [
            (key, self.model.bounds[key][0], self.model.bounds[key][1])
            for key in param_names
        ]

        best_parameter_list = self._run(bounds, **kwargs)
        if not best_parameter_list:
            if self.verbose:
                print(
                    f"The fit did not succeed with method {self.method}.")
            return None

        if 'unknown_param' in task_param:
            param_names = sorted(task_param['unknown_param'].keys())
        else:
            param_names = sorted(self.model.bounds.keys())

        self.best_param = {k: v for k, v in zip(param_names,
                                                best_parameter_list)}

        return {"best_param": self.best_param, "best_value": self.best_value}

    def _run(self, bounds, **kwargs):
        raise NotImplementedError
