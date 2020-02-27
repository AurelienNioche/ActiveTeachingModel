import os
import pickle
import numpy as np
import multiprocessing as mp
import scipy.optimize
from tqdm import tqdm
import matplotlib.pyplot as plt

# from user_data.models import User
# from teaching_material.selection import kanji

from user_data import analysis
from user_data.analysis.fit.scipy import DifferentialEvolution
# from analysis.fit.pygpgo import PyGPGO

from user_data.analysis.fit.learning_model.act_r.act_r import ActR
from user_data.analysis.fit.learning_model.act_r.custom import \
    ActRMeaning, ActRGraphic
from user_data.analysis.fit.learning_model.exponential_forgetting \
    import ExponentialForgetting, ExponentialForgettingAsymmetric
from user_data.analysis.fit.learning_model.rl import QLearner

# from teaching_material.selection import kanji, meaning

from user_data.analysis.fit.degenerate import Degenerate

from user_data.analysis.fit.comparison import bic

import user_data.analysis.tools.data
import user_data.analysis.similarity.graphic.measure
import user_data.analysis.similarity.semantic.measure
import user_data.analysis.plot.human

from user_data.settings import N_POSSIBLE_REPLIES, BKP_FIT

EPS = np.finfo(np.float).eps


FIG_FOLDER = os.path.join("fig", os.path.basename(__file__).split(".")[0])
os.makedirs(FIG_FOLDER, exist_ok=True)



class Fit:

    def __init__(self, hist, item, correct):

        self.hist = np.asarray(hist)
        bool_pres = self.hist == item
        self.n_pres = np.sum(bool_pres)
        t_pres = np.where(bool_pres)[0]
        self.delta = [t_pres[i+1] - t_pres[i] for i in range(len(t_pres)-1)]
        self.incorrect = np.invert(correct[bool_pres])

    def objective(self, param):
        alpha, beta = param
        # print("param", alpha, beta)
        p = np.zeros(self.n_pres)
        for i, delta in enumerate(self.delta):
            eta = alpha * (1 - beta)**i
            p_recall = np.exp(-eta*delta)
            p[i] = p_recall

        p[self.incorrect] = 1 - p[self.incorrect]

        r = - np.sum(np.log(p+EPS))
        return r

    def run(self):
        r = scipy.optimize.differential_evolution(
            self.objective, ((0., 10.), (0., 1.))
        )
        return r.x


def main(force=False):

    data = analysis.tools.data.get()

    # kanji = data["kanji"]
    # meaning = data["meaning"]

    bkp_unq = os.path.join("user_data", "bkp", "pilot_2019_09_02",
                           "unq_items.p")

    if not os.path.exists(bkp_unq):

        unq_items = []

        for i, user in enumerate(data["user_data"]):

            unq_items = list(np.unique(unq_items + list(user['hist'])))

        unq_items = sorted(unq_items)
        with open(bkp_unq,  "wb") as f:
            pickle.dump(unq_items, f)
    else:
        unq_items = pickle.load(open(bkp_unq, "rb"))

    bkp_fit = os.path.join("user_data", "bkp", "pilot_2019_09_02",
                           "fit_ind.p")
    if not os.path.exists(bkp_fit) or force:
        n_item = len(unq_items)
        n_user = len(data["user_data"])
        n_param = 2
        results = np.zeros((n_user, n_item, n_param))
        n = np.zeros((n_user, n_item))

        u_data = data["user_data"]
        for (i, user) in tqdm(enumerate(u_data), total=n_user):

            hist = user["hist"]
            correct = user["success"]

            for (j, item) in enumerate(unq_items):

                best_param = np.ones(n_param) * -1
                n_pres = np.sum(hist == item)
                if n_pres > 1:
                    f = Fit(hist=hist, item=item, correct=correct)
                    best_param[:] = f.run()
                # print(f"user {j} - item {i}: {best_param}")
                results[i, j] = best_param
                n[i, j] = n_pres

        with open(bkp_fit, "wb") as f:
            pickle.dump((results, n), f)

    else:
        results, n = pickle.load(open(bkp_fit, "rb"))

    n_user, n_item, n_param = results.shape
    print("n_user", n_user)

    fig, axes = plt.subplots(nrows=n_user, figsize=(6, 6*n_user))

    for i in range(n_user):
        relevant = n[i, :] > 1

        x_coord = results[i, relevant, 0]
        y_coord = results[i, relevant, 1]
        size = n[i, relevant] ** 2

        ax = axes[i]
        ax.scatter(x_coord, y_coord, alpha=0.5, s=size, color=f"C{i}")
        ax.set_xlim((0, 0.1))
    plt.savefig(os.path.join(FIG_FOLDER, "ind_plot.pdf"))



main()