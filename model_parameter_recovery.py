import multiprocessing
import os
import pickle
import scipy.stats

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from fit import fit
from learner.rl import QLearner
from learner.act_r import ActR
from learner.act_r_custom import ActRMeaning, ActRGraphic, ActRPlus

from simulation.data import SimulatedData
from simulation.task import Task

import plot.parameter_recovery


DATA_FOLDER = os.path.join("bkp", "model_evaluation")

os.makedirs(DATA_FOLDER, exist_ok=True)


class SimulationAndFit:

    def __init__(self, model, t_max=300, n_kanji=30, grades=(1, ),
                 normalize_similarity=False,
                 verbose=False, **kwargs):

        self.model = model

        self.tk = Task(t_max=t_max, n_kanji=n_kanji, grades=grades,
                       normalize_similarity=normalize_similarity,
                       verbose=verbose, generate_full_task=True)

        self.verbose = verbose

        self.kwargs = kwargs

    def __call__(self, seed):

        np.random.seed(seed)

        param = {}

        for bound in self.model.bounds:
            param[bound[0]] = np.random.uniform(bound[1], bound[2])

        data = SimulatedData(model=self.model, param=param, tk=self.tk,
                             verbose=self.verbose)
        f = fit.Fit(model=self.model, tk=self.tk, data=data,
                    **self.kwargs)
        fit_r = f.evaluate()
        return \
            {
                "initial": param,
                "recovered": fit_r["best_param"],
            }


def main(model, max_=20, t_max=300, n_kanji=30, normalize_similarity=True,
         force=False,
         **kwargs):

    extension = f'_{model.__name__}{model.version}_n{max_}_t{t_max}_' \
        f'k{n_kanji}_norm_{normalize_similarity}'

    file_path = os.path.join(DATA_FOLDER, f"parameter_recovery_{extension}.p")

    if not os.path.exists(file_path) or force:

        seeds = range(max_)

        pool = multiprocessing.Pool()
        results = list(tqdm(pool.imap_unordered(
            SimulationAndFit(model=model, t_max=t_max, n_kanji=n_kanji,
                             normalize_similarity=normalize_similarity,
                             **kwargs), seeds
        ), total=max_))

        r_keys = list(results[0].keys())
        param_names = results[0][r_keys[0]].keys()

        data = {
            pn:
                {
                    k: [] for k in r_keys
                }
            for pn in param_names
        }

        for r in results:
            for k in r_keys:
                for pn in param_names:
                    data[pn][k].append(r[k][pn])

        pickle.dump(data, open(file_path, 'wb'))

    else:
        data = pickle.load(open(file_path, 'rb'))

    plot.parameter_recovery.plot(data=data, extension=extension)


if __name__ == "__main__":

    main(ActRMeaning, max_=100, n_kanji=79, t_max=1000, force=False,
         normalize_similarity=True)