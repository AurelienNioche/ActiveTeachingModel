import os
import pickle
import numpy as np
import multiprocessing as mp
import scipy.optimize

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


class Fit:

    def __init__(self, hist, item):

        self.hist = np.asarray(hist)
        bool_pres = self.hist == item
        t_pres = np.where(bool_pres)[0]
        self.delta = [t_pres[i+1] - t_pres[i] for i in range(len(t_pres)-1)]

    def objective(self, param):
        alpha, beta = param

        p = np.zeros(self.delta)
        for i, delta in enumerate(self.delta):
            eta = alpha * (1 - beta)**i
            p[i] = np.exp(-eta*delta)
        return - np.sum(np.log(p+EPS))

    def run(self):
        return scipy.optimize.differential_evolution(
            self.objective, ((0, 1), (0, 1))
        ).x


def main():

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

    n_item = len(unq_items)
    n_user = len(data["user_data"])
    n_param = 2
    results = np.zeros((n_item, n_user, n_param))

    u_data = data["user_data"]
    for i, item in enumerate(unq_items):
        for j, user in enumerate(u_data):

            hist = user["hist"]
            if np.sum(hist == item) > 1:
                f = Fit(hist=hist, item=item)
                results[i, j] = f.run()
            else:
                results[i, j, :] = -1

    bkp = os.path.join("user_data", "bkp", "pilot_2019_09_02",
                           "fit_ind.p")
    with open(bkp, "wb") as f:
        pickle.dump(bkp, f)

main()