from GPyOpt.methods import BayesianOptimization

from fit.fit import Fit


class BayesianGPyOptFit(Fit):

    def __init__(self, tk, model, data, verbose=False, **kwargs):

        super().__init__(tk=tk, model=model, data=data, verbose=verbose,
                         method='BayesianGPyOpt',
                         **kwargs)

        # self.best_value = None
        self.best_param = None

    def evaluate(self, data=None, **kwargs):

        if data is not None:
            self.data = data

        domain = [
            {'name': f'{b[0]}',
             'type': 'continuous',
             'domain': (b[1], b[2])
             } for b in self.model.bounds
        ]

        myBopt = BayesianOptimization(f=self.objective,
                                      domain=domain)
        myBopt.run_optimization(max_iter=kwargs['max_iter'])

        best_param_list = myBopt.x_opt
        self.best_param = {b[0]: v for b, v in
                           zip(self.model.bounds,
                               best_param_list)}

        return self.get_stats()
