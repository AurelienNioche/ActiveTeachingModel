import numpy as np

from model.learner import QLearner, ActRLearner, ActRPlusLearner, ActRPlusPlusLearner

import scipy.optimize
from hyperopt import hp, fmin, tpe

import graphic_similarity.measure
import semantic_similarity.measure

IDX_PARAMETERS = 0
IDX_MODEL = 1
IDX_ARGUMENTS = 2


def _bic(lls, n, k):

    """
    :param lls: log-likelihood sum
    :param n: number of observations
    :param k: number of parameters
    :return: BIC
    """
    return -2 * lls + np.log(n) * k


def _log_likelihood_sum(p_choices):

    return np.sum(np.log(p_choices))

#
# def _objective(parameters, model, task, exp):
#
#     agent = model(parameters, task)
#     p_choices = agent.get_p_choices(exp)
#
#     if p_choices is None:
#         print("WARNING! Objective function returning 'None'")
#         return np.inf
#
#     return - _log_likelihood_sum(p_choices)


class Fit:

    def __init__(self, questions, replies, n_items, possible_replies, use_p_correct=False,
                 c_graphic=None, c_semantic=None):

        self.questions = questions
        self.replies = replies
        self.n_items = n_items
        self.possible_replies = possible_replies

        self.use_p_correct = use_p_correct

        self.c_graphic = c_graphic
        self.c_semantic = c_semantic

        self.t_max = len(self.questions)
        self.n_possible_replies = len(self.possible_replies[0])

    def _model_stats(self, p_choices, best_param):

        mean_p = np.mean(p_choices)

        lls = _log_likelihood_sum(p_choices)

        n = self.t_max
        k = len(best_param)
        bic = _bic(lls=lls, n=n, k=k)

        return mean_p, lls, bic

    def _evaluate(self, model, space):

        task, exp, fit_param = self._get_task_exp_fit_param()

        def objective(parameters):

            agent = model(parameters, task)
            p_choices_ = agent.get_p_choices(exp, fit_param)

            if p_choices_ is None:
                # print("WARNING! Objective function returning 'None'")
                to_return = np.inf

            else:
                to_return = - _log_likelihood_sum(p_choices_)

            # print(f"Objective returning: {to_return}")
            return to_return

        # res = scipy.optimize.differential_evolution(
        #     _objective, args=(model, task, exp),
        #     bounds=bounds)

        # res = scipy.optimize.minimize(
        #     x0=np.zeros(len(bounds)) * 0.5,
        #     fun=_objective, args=(model, task, exp),
        #     bounds=bounds)

        best_param = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10**3)

        # print(best_param)
        # best_param = res.x
        # success = res.success
        # print(f"Fit was successful: {success}")

        # Get probabilities with best param
        learner = model(best_param, task)
        p_choices = learner.get_p_choices(exp, fit_param)

        # Compute bic, etc.
        mean_p, lls, bic = self._model_stats(p_choices=p_choices, best_param=best_param)

        self._print(model.__name__, best_param, mean_p, lls, bic)
        return mean_p, bic

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

        space = [
            hp.uniform('alpha', 0, 1),
            hp.uniform('tau', 0.002, 0.5)
        ]

        model = QLearner
        # bounds = [(0.00, 1.00), (0.002, 0.5)]

        return self._evaluate(model=model, space=space)

    def act_r(self):

        model = ActRLearner
        # bounds = (
        #     (0, 1.0),  # d
        #     (0.00, 5),  # tau
        #     (0.0000001, 10)  # s
        # )

        space = (
            hp.uniform('d', 0, 1.0),  # d
            hp.uniform('tau', 0.00, 5),  # tau
            hp.uniform('s', 0.0000001, 10)  # s
        )

        return self._evaluate(model=model, space=space)

    def act_r_plus(self):

        model = ActRPlusLearner
        # bounds = (
        #     (0, 1.0),  # d
        #     (0.00, 10),  # tau
        #     (0.0000001, 10),  # s
        #     (-10, 10),  # g
        #     (-10, 10))  # m

        space = (
            hp.uniform('d', 0, 1.0),  # d
            hp.uniform('tau', 0.00, 5),  # tau
            hp.uniform('s', 0.0000001, 10),  # s
            hp.uniform('g', -10, 10),
            hp.uniform('m', -10, 10),
        )

        return self._evaluate(model=model, space=space)

    def act_r_plus_plus(self):

        model = ActRPlusPlusLearner
        # bounds = (
        #     (0, 1.0),  # d
        #     (0.00, 10),  # tau
        #     (0.0000001, 10),  # s
        #     (-1, 1),  # g
        #     (-1, 1),  # m
        #     (0, 1),  # g_mu
        #     (0.01, 5),  # g_sigma
        #     (0, 1),  # s_mu
        #     (0.01, 5))  # s_sigma

        space = (
            hp.uniform('d', 0, 1.0),  # d
            hp.uniform('tau', 0.00, 5),  # tau
            hp.uniform('s', 0.0000001, 10),  # s
            hp.uniform('g', -10, 10),
            hp.uniform('m', -10, 10),
            hp.uniform('g_mu', 0, 1),  # g_mu
            hp.uniform('g_sigma', 0.01, 5),  # g_sigma
            hp.uniform('m_mu', 0, 1),  # s_mu
            hp.uniform('m_sigma', 0.01, 5)  # s_sigma
        )

        return self._evaluate(model=model, space=space)

    @classmethod
    def _print(cls, model_name, best_param, mean_p, lls, bic):

        if type(best_param) == dict:
            best_param = {k: f"{v:.3f}" for k, v in best_param.items()}
            dis_best_param = f"{best_param}, "
        else:
            dis_best_param = f'{[f"{i:.3f}" for i in best_param]}, '

        print(f'[{model_name}] Best param: ' + dis_best_param +
              f"LLS: {lls:.2f}, " +
              f'BIC: {bic:.2f}, mean(P): {mean_p:.3f}')
