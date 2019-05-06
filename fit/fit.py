import numpy as np

from learner.rl import QLearner
from learner.act_r import ActR
from learner.act_r_custom import ActRMeaning, ActRGraphic, ActRPlus, ActRPlusPlus

import scipy.optimize
from hyperopt import hp, fmin, tpe

IDX_PARAMETERS = 0
IDX_MODEL = 1
IDX_ARGUMENTS = 2

MAX_EVALS = 500  # Only if tpe


class Fit:

    def __init__(self, questions, replies, n_items, possible_replies, method='tpe', use_p_correct=False,
                 c_graphic=None, c_semantic=None):

        self.questions = questions
        self.replies = replies
        self.n_items = n_items
        self.possible_replies = possible_replies

        self.use_p_correct = use_p_correct
        self.method = method

        self.c_graphic = c_graphic
        self.c_semantic = c_semantic

        self.t_max = len(self.questions)
        self.n_possible_replies = len(self.possible_replies[0])

    def _bic(self, lls, k):
        """
        :param lls: log-likelihood sum
        :param k: number of parameters
        :return: BIC
        """
        return -2 * lls + np.log(self.t_max) * k

    @classmethod
    def _log_likelihood_sum(cls, p_choices):
        return np.sum(np.log(p_choices))

    def _model_stats(self, p_choices, best_param):

        mean_p = np.mean(p_choices)
        lls = self._log_likelihood_sum(p_choices)
        bic = self._bic(lls=lls, k=len(best_param))

        return mean_p, lls, bic

    def _evaluate(self, model, bounds):

        task, exp, fit_param = self._get_task_exp_fit_param()

        def objective(parameters):

            agent = model(parameters=parameters, task_features=task)
            p_choices_ = agent.get_p_choices(exp=exp, fit_param=fit_param)

            if p_choices_ is None or np.any(np.isnan(p_choices_)):
                # print("WARNING! Objective function returning 'None'")
                to_return = np.inf

            else:
                to_return = - self._log_likelihood_sum(p_choices_)

            return to_return

        if self.method == 'tpe':  # Tree of Parzen Estimators

            space = [hp.uniform(*b) for b in bounds]
            best_param = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS)

        elif self.method in ('de', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP'):  # Differential evolution

            bounds_scipy = [b[-2:] for b in bounds]

            print("Finding best parameters...", end=' ')

            if self.method == 'de':
                res = scipy.optimize.differential_evolution(
                        func=objective, bounds=bounds_scipy)
            else:
                x0 = np.zeros(len(bounds)) * 0.5
                res = scipy.optimize.minimize(fun=objective, bounds=bounds_scipy, x0=x0)

            best_param = {b[0]: v for b, v in zip(bounds, res.x)}
            print(f"{res.message} [best loss: {res.fun}]")
            if not res.success:
                raise Exception(f"The fit did not succeed with method {self.method}.")

        else:
            raise Exception(f'Method {self.method} is not defined')

        # Get probabilities with best param
        learner = model(parameters=best_param, task_features=task)
        p_choices = learner.get_p_choices(exp=exp, fit_param=fit_param)

        # Compute bic, etc.
        mean_p, lls, bic = self._model_stats(p_choices=p_choices, best_param=best_param)

        self._print(model.__name__, best_param, mean_p, lls, bic)
        return best_param, mean_p, lls, bic

    def _get_task_exp_fit_param(self):

        task = {
            'n_items': self.n_items,
            't_max': self.t_max,
            'n_possible_replies': self.n_possible_replies,
            'c_graphic': self.c_graphic,
            'c_semantic': self.c_semantic
        }

        exp = {
            'questions': self.questions,
            'replies': self.replies,
            'possible_replies': self.possible_replies
        }

        fit_param = {
            'use_p_correct': self.use_p_correct
        }

        return task, exp, fit_param

    def rl(self):

        bounds = (
            ('alpha', 0, 1),
            ('tau', 0.002, 0.5)
        )
        model = QLearner

        return self._evaluate(model=model, bounds=bounds)

    def act_r(self):

        model = ActR
        bounds = (
            ('d', 0, 1.0),
            ('tau', -5, 5),
            ('s', 0.0000001, 1)
        )

        return self._evaluate(model=model, bounds=bounds)

    def act_r_meaning(self):

        model = ActRMeaning
        bounds = (
            ('d', 0.0000001, 1.0), 
            ('tau', -5, 5),
            ('s', 0.0000001, 1),
            ('m', -2, 2),
        )

        return self._evaluate(model=model, bounds=bounds)
    
    def act_r_graphic(self):

        model = ActRGraphic
        bounds = (
            ('d', 0.0000001, 1.0), 
            ('tau', -5, 5),
            ('s', 0.0000001, 1), 
            ('g', -2, 2),
        )

        return self._evaluate(model=model, bounds=bounds)
    
    def act_r_plus(self):

        model = ActRPlus
        bounds = (
            ('d', 0, 1.0),
            ('tau', 0.00, 5),
            ('s', 0.0000001, 10),
            ('g', -10, 10),
            ('m', -10, 10),
        )

        return self._evaluate(model=model, bounds=bounds)

    def act_r_plus_plus(self):

        model = ActRPlusPlus
        bounds = (
            ('d', 0, 1.0),
            ('tau', 0.00, 5),
            ('s', 0.0000001, 10),
            ('g', -10, 10),
            ('m', -10, 10),
            ('g_mu', 0, 1),
            ('g_sigma', 0.01, 5),
            ('m_mu', 0, 1),
            ('m_sigma', 0.01, 5)
        )

        return self._evaluate(model=model, bounds=bounds)

    def _print(self, model_name, best_param, mean_p, lls, bic):

        dsp_use_p_correct = "p_correct" if self.use_p_correct else "p_choice"
        dsp_best_param = ''.join(f'{k}={round(best_param[k], 3)}, ' for k in sorted(best_param.keys()))

        print(f"[{model_name} - '{self.method}' - {dsp_use_p_correct}] Best param: " + dsp_best_param +
              f"LLS: {round(lls, 2)}, " +
              f'BIC: {round(bic, 2)}, mean(P): {round(mean_p, 3)}\n')
