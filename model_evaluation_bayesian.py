# import multiprocessing
import os
# import sys
#
# import pickle
# import scipy.stats
#
import matplotlib.pyplot as plt
import numpy as np
# from tqdm import tqdm
#
from fit.fit import Fit
from fit.bayesian import BayesianFit
from fit.bayesian_gpyopt import BayesianGPyOptFit
from fit.bayesian_pygpgo import BayesianPYGPGOFit
# from learner.rl import QLearner
from learner.act_r import ActR
from learner.act_r_custom import ActRMeaning, ActRGraphic, ActRPlus
#
from simulation.data import SimulatedData
from simulation.task import Task

from fit.demos.bayesian import run_test

from datetime import timedelta
from time import time

DATA_FOLDER = os.path.join("bkp", "model_evaluation")
FIG_FOLDER = "fig"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)


def main():

    param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02}
    model = ActRMeaning

    # param = {"d": 0.5, "tau": 0.01, "s": 0.06}
    # model = ActR

    t_max = 500
    n_item = 79
    grades = (1, )
    normalize_similarity = True

    init_evals = 3
    max_iter = 50 - init_evals

    verbose = True

    tk = Task(t_max=t_max, n_kanji=n_item, grades=grades,
              normalize_similarity=normalize_similarity,
              generate_full_task=True)

    data = SimulatedData(model=model, param=param, tk=tk)

    # print("Differential evolution")
    # f = Fit(model=model, tk=tk, data=data)
    # print(f.evaluate(maxiter=30, workers=-1))
    # print()
    # print("BayesianOpt")
    # f = BayesianFit(model=model, tk=tk, data=data)
    # print(f.evaluate(verbose=2, init_points=0, n_iter=30))
    # print()
    # print("BayesianGPyOpt")
    # f = BayesianGPyOptFit(model=model, tk=tk, data=data)
    # print(f.evaluate(max_iter=30))
    # print()
    t0 = time()
    print("BayesianPYGPGO")
    f = BayesianPYGPGOFit(model=model, tk=tk, data=data)

    obj_with_true_param = f.objective(keep_in_history=False, **param)
    print('obj with true param', obj_with_true_param)

    f.evaluate(max_iter=max_iter, init_evals=init_evals, verbose=verbose)

    # print(f.evaluate(n_iter=max_iter, init_points=init_evals))

    print("Time:", timedelta(seconds=time() - t0))

    n_param = len(param)

    param_key = sorted(list(param.keys()))

    history = f.history
    objective = f.obj_values
    fig, axes = plt.subplots(nrows=n_param+1)

    x = np.arange(len(history))

    for i in range(n_param):
        ax = axes[i]

        key = param_key[i]

        y = [v[key] for v in history]
        ax.plot(x, y, color=f"C{i}")
        ax.axhline(param[key], linestyle='--', color="black", alpha=0.5,
                   zorder=-1)
        ax.set_ylabel(key)

        ax.set_xlim(min(x), max(x))

        ax.set_xticks([])

    ax = axes[-1]

    ax.set_xlim(min(x), max(x))
    ax.plot(x, objective)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Objective")

    ax.axhline(obj_with_true_param, linestyle='--', color="black", alpha=0.5,
               zorder=-1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    main()
