import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
import os
from scipy.optimize import minimize, differential_evolution

FIG_FOLDER = os.path.join("fig", os.path.basename(__file__).split(".")[0])
os.makedirs(FIG_FOLDER, exist_ok=True)


N_ITEM = 100

ALPHA = 0.04
BETA = 0.2

N_ITER = 2000

EPSILON = 0.2


def compute_next_recall(hist, item, tau):
    n_view = hist.count(item)
    return - np.log(tau) / eta(ALPHA, BETA, n_view)


def objective(tau, new_strategy=True):

    hist = []

    dumb_value = 999999
    next_recall = np.ones(N_ITEM) * dumb_value

    for t in range(N_ITER):

        if t == 0:
            item = 0

        else:
            to_r = np.where(t >= next_recall[:])[0]
            if len(to_r):
                item = np.argmin(next_recall)
            else:
                to_select = np.where(next_recall[:] == dumb_value)[0]
                if len(to_select):
                    item = to_select[0]
                else:
                    item = np.argmin(next_recall)

        hist.append(item)
        next_recall[item] = t + compute_next_recall(hist, item, tau)

        if new_strategy is True:

            ps = np.zeros(N_ITEM)
            for i in range(N_ITEM):
                ps[i] = exp_forget(i, ALPHA, BETA, hist)

            if np.all(ps[:] > (1 - EPSILON)):
                return t

    if new_strategy is True:
        return N_ITER

    else:
        ps = np.zeros(N_ITEM)
        for i in range(N_ITEM):
            ps[i] = exp_forget(i, ALPHA, BETA, hist)
        return np.sum(ps[:] > (1-EPSILON))

    # return N_ITER


def fig_memory(y, pres=None, f_name='memory.pdf'):

    fig, ax = plt.subplots(figsize=(4, 4))

    if pres is not None:
        for pr in pres:
            ax.axvline(pr+0.5, linestyle="--", color="0.2",
                       linewidth=0.5)
    ax.plot(y)
    # ax.axhline(0.9, linestyle="--", color="0.2")
    ax.set_ylim(0., 1.1)
    ax.set_xlim(0, len(y))

    ax.set_xlabel("Time")
    ax.set_ylabel("Probability of recall")

    plt.tight_layout()
    plt.show()
    #fig_folder = os.path.join("fig", "illustration")
    #os.makedirs(fig_folder, exist_ok=True)
    # plt.savefig(os.path.join(fig_folder, f_name))


def eta(alpha, beta, n_view):
    return alpha * (1 - beta) ** (n_view - 1)


def exp_forget(i, alpha, beta, hist):

    hist_array = np.asarray(hist)
    t = len(hist_array)

    bool_view = hist_array[:] == i
    n_view = np.sum(bool_view)
    if n_view == 0:
        return 0.0
    delta = t - np.max(np.arange(t)[bool_view])
    eta = alpha * (1 - beta)**(n_view-1)
    return np.exp(-eta*delta)


def exp():

    alpha, beta = 0.02, 0.2,

    tau = 0.9
    a = [- np.log(tau) / (alpha*(1-beta)**i) for i in range(5)]
    pres = [0, ]
    for a_ in a:
        pres.append(int(pres[-1] + a_))
    print(pres)

    t_max = int(max(pres)) + 1

    hist = np.ones(t_max)
    hist[np.asarray(pres)] = 0
    ps = []
    for t in range(t_max):
        p = exp_forget(0, alpha, beta, hist[:t])
        ps.append(p)
    fig_memory(ps)


def p_recall_as_a_function_of_time(tau):

    hist = []

    dumb_value = 999999
    next_recall = np.ones(N_ITEM) * dumb_value

    seen = np.zeros(N_ITEM, dtype=bool)

    items = np.arange(N_ITEM)

    ps = np.zeros((N_ITEM, N_ITER))

    for t in range(N_ITER):

        if t == 0:
            item = 0

        else:
            to_r = np.where(t >= next_recall[:])[0]
            if len(to_r):
                item = np.argmin(next_recall)
            else:
                to_select = np.where(next_recall[:] == dumb_value)[0]
                if len(to_select):
                    item = to_select[0]
                else:
                    item = np.argmin(next_recall)

        seen[item] = True
        hist.append(item)
        next_recall[item] = t + compute_next_recall(hist, item, tau)

        for i in items[seen]:
            ps[i, t] = exp_forget(i, ALPHA, BETA, hist)
    # ps = np.zeros(N_ITEM)
    # for i in range(N_ITEM):
    #     ps[i] = exp_forget(i, ALPHA, BETA, hist)

    # if np.all(ps[:] > (1-EPSILON)):
    #     return t

    fig, ax = plt.subplots()
    for i in items[seen]:
        ax.plot(ps[i], linewidth=0.2, alpha=0.5, color='C0')

    plt.savefig(os.path.join(FIG_FOLDER, f"play_tau_{tau}.pdf"))

    return np.sum(ps[:] > (1-EPSILON))


def optimize_using_grid_explo():

    x = np.linspace(0.01, 0.99, 10)

    with mp.Pool() as pool:
        y = list(tqdm(pool.imap(objective, x), total=len(x)))

    # y = pool.map(objective, x)
    #

    plt.plot(x, y)
    plt.savefig(os.path.join(FIG_FOLDER, "exp_new_teacher.pdf"))


    # r = differential_evolution(objective, bounds=[(0, 1)])
    # print(r)

if __name__ == "__main__":
    optimize_using_grid_explo()