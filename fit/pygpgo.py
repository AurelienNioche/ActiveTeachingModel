from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO

from . abstract_class import Fit
from . objective import objective


class PyGPGO(Fit):

    def __init__(self, **kwargs):

        super().__init__(method='pygpgo', **kwargs)

    def objective(self, keep_in_history=True, **param):

        value = super().objective(param=param, keep_in_history=keep_in_history)
        return - value

    def _run(self, **kwargs):

        """
        kwargs: for GPGO: n_jobs, for run: max_iter, init_evals
        """

        sexp = squaredExponential()
        gp = GaussianProcess(sexp)
        acq = Acquisition(mode='ExpectedImprovement')
        f = self.objective
        param = {
            f'{b[0]}': ('cont', [b[1], b[2]])
            for b in self.model.bounds
        }

        # surrogate = BoostedTrees()

        opt = GPGO(gp, acq, f, param, **kwargs)

        opt.run()

        r = opt.getResult()

        self.best_param = dict(r[0])
        self.best_value = r[1]

        return True
