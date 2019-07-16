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


DATA_FOLDER = os.path.join("bkp", "model_evaluation")
FIG_FOLDER = "fig"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)


class SimulationAndFit:

    def __init__(self, model, t_max=300, n_kanji=30, grade=1,
                 normalize_similarity=False,
                 verbose=False, **kwargs):

        self.model = model

        self.tk = Task(t_max=t_max, n_kanji=n_kanji, grade=grade,
                       normalize_similarity=normalize_similarity,
                       verbose=verbose)

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


def create_fig(data, extension=''):

    # Create fig
    fig, axes = plt.subplots(nrows=len(data.keys()), figsize=(5, 10))

    i = 0
    for title, dic in sorted(data.items()):

        ax = axes[i]

        x, y = dic["initial"], dic["recovered"]

        ax.scatter(x, y, alpha=0.5)

        cor, p = scipy.stats.pearsonr(x, y)

        print(f'Pearson corr {title}: $r_pearson={cor:.2f}$, $p={p:.3f}$')

        ax.set_title(title)

        max_ = max(x+y)
        min_ = min(x+y)

        ax.set_xlim(min_, max_)
        ax.set_ylim(min_, max_)

        ticks_positions = [round(i, 2) for i in np.linspace(min_, max_, 3)]

        ax.set_xticks(ticks_positions)
        ax.set_yticks(ticks_positions)

        ax.set_aspect(1)
        i += 1

    plt.tight_layout()
    f_name = f"parameter_recovery{extension}.pdf"
    fig_path = os.path.join(FIG_FOLDER, f_name)
    plt.savefig(fig_path)
    print(f"Figure '{fig_path}' created.\n")
    plt.tight_layout()


def main(model, max_=20, t_max=300, n_kanji=30, normalize_similarity=True,
         force=False,
         **kwargs):

    extension = f'_{model.__name__}{model.version}_n{max_}_t{t_max}_k{n_kanji}'

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


if __name__ == "__main__":

    main(ActR, max_=100, n_kanji=79, t_max=100, force=True)
