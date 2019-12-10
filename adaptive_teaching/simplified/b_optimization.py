import numpy as np
import GPyOpt



EPS = np.finfo(np.float).eps


class BaysesianOptimizer:

    def __init__(self, bounds, initial_design_numdata=5):

        bounds = [
            {'name': f'tamere',
             'type': 'continuous',
             'domain': (b[0], b[1])
             } for b in bounds
        ]

        self.opt = GPyOpt.methods.BayesianOptimization(
            f=self.objective, domain=bounds, acquisition_type='EI',
            exact_feval=False,
            initial_design_numdata=initial_design_numdata,
            normalize_Y=False)

    def objective(self, param):

        param = param[0]
        x, y = param

        return 10**2 + x**2 + 2*y + 0.1 + 0.1 * np.random.randn()

    def run(self, max_iter=np.inf, max_time=5):

        self.opt.run_optimization(
            max_iter=max_iter,
            max_time=max_time,
            eps=EPS)

    def best_value(self):
        return self.opt.fx_opt

    def best_param(self):
        return self.opt.x_opt

    def std(self):

        n_param = len(self.opt.x_opt)

        model = self.opt.model.model
        eval_points = np.zeros((1, n_param))
        eval_points[0, :] = self.opt.x_opt
        m, v = model.predict(eval_points)
        print(m, v)
        estimated_var = v[0][0]

        return np.sqrt(estimated_var)


def main():

    opt = BaysesianOptimizer(bounds=((0, 1), (0, 1)),
                             initial_design_numdata=100)
    opt.run(max_iter=5)

    best_param = opt.best_param()
    std = opt.std()

    print(best_param)
    print(std)

main()

