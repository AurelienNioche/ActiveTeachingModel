import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

FIG_FOLDER = os.path.join("fig", os.path.basename(__file__).split(".")[0])
os.makedirs(FIG_FOLDER, exist_ok=True)

N_ITEM = 20
N_ITER = 10

ALPHA = 0.04
BETA = 0.2

EPS = np.finfo(np.float).eps


def exp_forget(i, alpha, beta, hist):
    hist_array = np.asarray(hist)
    t = len(hist_array)

    bool_view = hist_array[:] == i
    n_view = np.sum(bool_view)
    if n_view == 0:
        return 0.0
    delta = t - np.max(np.arange(t)[bool_view])
    eta = alpha * (1 - beta) ** (n_view - 1)
    return np.exp(-eta * delta)


def main():
    hist = []

    seen = np.zeros(N_ITEM, dtype=bool)

    items = np.arange(N_ITEM)

    bkp_p = np.zeros((N_ITEM, N_ITER))

    for t in tqdm(range(N_ITER)):

        if t == 0:
            item = 0

        else:
            n_seen = np.sum(seen)

            seen_item = items[seen]

            n_seen_plus_one = n_seen + 1

            baseline = np.zeros(n_seen_plus_one)
            for i, x in enumerate(seen_item):
                p = exp_forget(x, ALPHA, BETA, hist)
                baseline[i] = p
                bkp_p[x, t - 1] = p

            after = np.zeros((n_seen_plus_one, n_seen_plus_one))
            hist_plus_dummy = hist + [-1, ]
            for i, x in enumerate(seen_item):
                after[:, i] = exp_forget(x, ALPHA, BETA, hist_plus_dummy)

            for i, x in enumerate(seen_item):
                hist_plus_x = hist + [x, ]
                after[i, i] = exp_forget(x, ALPHA, BETA, hist_plus_x)

            baseline += EPS
            after += EPS

            ent_base_line = - np.sum(baseline * np.log(baseline))

            print("ent base line", ent_base_line)
            ent_plus = np.zeros(n_seen_plus_one)

            for i in range(n_seen_plus_one):
                ent_plus[i] = - np.sum(after[i, :] * np.log(after[i, :]))

            diff_ent = ent_plus - ent_base_line

            idx = np.argmax(diff_ent)
            print("ent", diff_ent)
            if idx == n_seen:
                item = items[n_seen]
            else:
                item = items[seen][idx]

        hist.append(item)
        print("hist", hist, "t", t)
        seen[item] = 1

    fig, ax = plt.subplots()
    for x in items[seen]:
        ax.plot(bkp_p[x, :], linewidth=0.2, alpha=0.5, color='C0')

    plt.savefig(os.path.join(FIG_FOLDER, f"info_theory.pdf"))


if __name__ == "__main__":
    main()
