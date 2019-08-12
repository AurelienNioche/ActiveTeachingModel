from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.surrogates.BoostedTrees import BoostedTrees
from pyGPGO.GPGO import GPGO, EventLogger

import multiprocessing as mp

import numpy as np
from fit.fit import Fit


def objective(model, tk, data, param, show=False):

    if show:
        print('\n')

    agent = model(param=param, tk=tk)
    t_max = data.t_max
    diff = np.zeros(t_max)
    # p_choices_ = agent.get_p_choices(data=data,
    #                                  stop_if_zero=False,
    #                                  use_p_correct=True)

    for t in range(t_max):
        item = data.questions[t]
        if show:
            print("Item", item)
        p_r = agent.p_recall(item=item)
        if show:
            print('p_recall: ', p_r)
        s = data.success[t]
        if show:
            print("success: ", s)
        # if s:
        #     p_choice = p_r
        # else:
        #     p_choice = 1-p_r

        # if show:
        #     print('p_model:', p_choice)
        diff[t] = (s - p_r)

        agent.learn(item)

    # diff[diff < 0] *= 1.1
    diff = np.power(diff, 2)
    value = -np.sum(diff)
    if show:
        print("total value", value)
        print()
    return value


class MyGPGO(GPGO):

    # noinspection PyMissingConstructor
    def __init__(self, surrogate, acquisition, f, parameter_dict, n_jobs=1,
                 verbose=True):

        # super().__init__(surrogate, acquisition, f, parameter_dict, n_jobs)

        self.GP = surrogate
        self.A = acquisition
        self.f = f
        self.parameters = parameter_dict
        self.n_jobs = n_jobs

        self.parameter_key = list(parameter_dict.keys())
        self.parameter_value = list(parameter_dict.values())
        self.parameter_type = [p[0] for p in self.parameter_value]
        self.parameter_range = [p[1] for p in self.parameter_value]

        self.history = []

        self.verbose = verbose
        if verbose:
            self.logger = EventLogger(self)

    def run(self, max_iter=10, init_evals=3, resume=False, init_param=None):
        """
        Runs the Bayesian Optimization procedure.

        Parameters
        ----------
        max_iter: int
            Number of iterations to run. Default is 10.
        init_evals: int
            Initial function evaluations before fitting a GP. Default is 3.
        resume: bool
            Whether to resume the optimization procedure from the last evaluation. Default is `False`.

        """
        if not resume:
            self.init_evals = init_evals
            self._firstRun(n_eval=self.init_evals, init_param=init_param)
            if self.verbose:
                self.logger._printInit(self)
        for iteration in range(max_iter):
            self._optimizeAcq()
            self.updateGP()
            if self.verbose:
                self.logger._printCurrent(self)

    def _firstRun(self, n_eval=3, init_param=None):
        """
        Performs initial evaluations before fitting GP.

        Parameters
        ----------
        n_eval: int
            Number of initial evaluations to perform. Default is 3.

        """
        self.X = np.empty((n_eval, len(self.parameter_key)))
        self.y = np.empty((n_eval,))

        if init_param is not None:
            s_param = init_param
            s_param_val = [s_param[k] for k in self.parameter_key]
            self.X[0] = s_param_val
            self.y[0] = self.f(**s_param)

            iterator = range(1, n_eval)

        else:
            iterator = range(0, n_eval)

        for i in iterator:
            s_param = self._sampleParam()
            s_param_val = list(s_param.values())
            self.X[i] = s_param_val
            self.y[i] = self.f(**s_param)

        self.GP.fit(self.X, self.y)
        self.tau = np.max(self.y)
        self.history.append(self.tau)


class BayesianPYGPGOFit:

    def __init__(self, verbose=False, seed=123,
                 n_jobs=mp.cpu_count(),
                 **kwargs):

        # super().__init__(tk=tk, model=model, data=data, verbose=verbose,
        #                  method='BayesianGPyOpt',
        #                  **kwargs)

        self.best_value = None
        self.best_param = None

        self.history_eval_param = []
        self.obj_values = []

        self.n_jobs = n_jobs

        self.history_best_fit_param = []
        self.history_best_fit_value = []

        self.time_out = None

        self.verbose = verbose

        self.opt = None

        np.random.seed(seed)

        self.model = None
        self.tk = None
        self.data = None

    def objective(self, **param):

        return objective(model=self.model, data=self.data, tk=self.tk,
                         param=param)

    def evaluate(self, model, tk, data, **kwargs):

        self.model = model
        self.tk = tk
        self.data = data

        self.history_eval_param = []

        param = {
            f'{b[0]}': ('cont', [b[1], b[2]])
            for b in self.model.bounds
        }

        f = self.objective

        sexp = squaredExponential()
        gp = GaussianProcess(sexp)

        # surrogate = BoostedTrees()
        acq = Acquisition(mode='ExpectedImprovement')

        opt = MyGPGO(gp, acq, f, param,
                     n_jobs=self.n_jobs,
                     verbose=self.verbose)
        if self.best_param is None:
            if self.verbose:
                print("No previous best param for now")
            self.best_param = {}
            opt.run(**kwargs)

        else:
            if self.verbose:
                print(f"Best param is {self.best_param}")
            opt.run(init_param=self.best_param, **kwargs)

        r = opt.getResult()

        self.best_param = dict(r[0])
        self.best_value = r[1]

        self.history_best_fit_param.append(self.best_param)
        self.history_best_fit_value.append(self.best_value)

        return self.best_param
