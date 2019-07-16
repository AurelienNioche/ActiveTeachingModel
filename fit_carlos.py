import os

import numpy as np
from tqdm import tqdm

from fit import fit
from learner.act_r import ActR
from simulation.data import SimulatedData, Data
from simulation.task import Task

MAX_EVAL = 500  # Only if tpe
DATA_FOLDER = os.path.join("bkp", "model_evaluation")


def main():
    n_param = 3
    t_min = 25
    t_max = 300
    n_kanji = 79
    grade = 1
    normalize_similarity = False
    verbose = False
    model = ActR
    tk = Task(t_max=t_max, n_kanji=n_kanji, grade=grade,
              normalize_similarity=normalize_similarity,
              verbose=verbose)

    np.random.seed(123)

    param = {}

    for bound in model.bounds:
        param[bound[0]] = np.random.uniform(bound[1], bound[2])

    data = SimulatedData(model=ActR, param=param, tk=tk,
                         verbose=False)

    changes = np.zeros((n_param, t_max - t_min - 1))

    data_view = Data(n_items=n_kanji, questions=data.questions[:t_min],
                     replies=data.replies[:t_min],
                     possible_replies=data.possible_replies[:t_min, :])
    tk.t_max = t_min

    f = fit.Fit(model=ActR, tk=tk, data=data_view)
    fit_r = f.evaluate()

    best_v = {}
    best_v.update(fit_r["best_param"])

    i = 0
    for t in tqdm(range(t_min+1, t_max)):

        data_view = Data(n_items=n_kanji, questions=data.questions[:t],
                         replies=data.replies[:t],
                         possible_replies=data.possible_replies[:t, :])

        tk.t_max = t

        f = fit.Fit(model=ActR, tk=tk, data=data_view)
        fit_r = f.evaluate()

        for j, k in enumerate(sorted(fit_r["best_param"].keys())):
            changes[j, i] = np.abs(best_v[k] - fit_r["best_param"][k])

        best_v.update(fit_r["best_param"])

        i += 1


# Suppose you have an array 'data' with each line being the values for
# a specific agent, and each column, values for a fit including data until
# specific time step.
#
# mean = np.mean(data, axis=0)
# sem = scipy.stats.sem(data, axis=0)
# ax.plot(mean, lw=1.5)
# ax.fill_between(
#     range(len(mean)),
#     y1=mean - sem,
#     y2=mean + sem,
#     alpha=0.2)


if __name__ == "__main__":
    main()
