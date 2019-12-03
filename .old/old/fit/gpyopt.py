import numpy as np

from . abstract_class import Fit
from . objective import objective

import GPyOpt


class Gpyopt(Fit):

    def __init__(self, **kwargs):

        super().__init__(method='de', **kwargs)

        self.opt = None

    def objective(self, param, keep_in_history=True):

        param = param[0]

        return super().objective(param=param, keep_in_history=keep_in_history)

    def _run(
            self,
            bounds,
            initial_design_numdata=10,
            max_iter=np.inf,
            max_time=5,
            **kwargs):

        bounds = [
            {'name': f'{b[0]}',
             'type': 'continuous',
             'domain': (b[1], b[2])
             } for b in bounds
        ]

        eps = 10e-6

        myBopt = GPyOpt.methods.BayesianOptimization(
            f=self.objective, domain=bounds, acquisition_type='EI',
            exact_feval=False,
            initial_design_numdata=initial_design_numdata,
            normalize_Y=False)

        myBopt.run_optimization(max_iter=max_iter,
                                max_time=max_time,
                                eps=eps)

        best_param_list = myBopt.x_opt
        # self.best_param = {b[0]: v for b, v in
        #                    zip(self.model.bounds,
        #                        best_param_list)}

        self.best_value = myBopt.fx_opt

        self.opt = myBopt

        return best_param_list

    def get_estimated_var(self, n_unknown_param=None):

        if not n_unknown_param:
            n_unknown_param = len(self.model.bounds)

        model = self.opt.model.model
        eval_points = np.zeros((1, n_unknown_param))
        eval_points[0, :] = self.opt.x_opt
        m, v = model.predict(eval_points)
        estimated_var = v[0][0]

        return estimated_var
