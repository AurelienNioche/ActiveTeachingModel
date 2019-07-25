import multiprocessing
import os
import pickle
import scipy.stats

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from fit.bayesian import BayesianFit
from learner.rl import QLearner
from learner.act_r import ActR
from learner.act_r_custom import ActRMeaning, ActRGraphic, ActRPlus

from simulation.data import SimulatedData
from simulation.task import Task


DATA_FOLDER = os.path.join("bkp", "model_evaluation")
FIG_FOLDER = "fig"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(FIG_FOLDER, exist_ok=True)

def main():

    param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02}
    model = ActRMeaning

    t_max = 300
    n_item = 30
    grades = (1, )
    normalize_similarity = True

    tk = Task(t_max=t_max, n_kanji=n_item, grades=grades,
              normalize_similarity=normalize_similarity,
              generate_full_task=True)

    data = SimulatedData(model=model, param=param, tk=tk)
    f = BayesianFit(model=model, tk=tk, data=data)
    best_param = f.evaluate(verbose=2)
    print(best_param)


if __name__ == "__main__":

    main()
