import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import minimize


N_ITEM = 100

ALPHA = 0.02
BETA = 0.2

N_ITER = 1000


def compute_next_recall(hist, item, tau):
    n_view = hist.count(item)
    return - np.log(tau)/ eta(ALPHA, BETA, n_view)


def objective(tau):

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

    ps = np.zeros(N_ITEM)
    for i in range(N_ITEM):
        ps[i] = exp_forget(i, ALPHA, BETA, hist)

    return np.mean(ps)


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
    alpha * (1 - beta) ** (n_view - 1)


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


def main():

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


if __name__ == "__main__":
    main()
