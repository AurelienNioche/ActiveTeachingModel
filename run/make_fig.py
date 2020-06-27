import os

import numpy as np
from tqdm import tqdm

from teacher.psychologist import Psychologist

from plot.plot import DataFig, plot

SCRIPT_NAME = os.path.basename(__file__).split(".")[0]

FIG_FOLDER = os.path.join("fig", SCRIPT_NAME)
os.makedirs(FIG_FOLDER, exist_ok=True)


def _format_parameter_recovery(param_recovery_cond, tk):

    n_obs = tk.terminal_t
    n_param = len(tk.param)

    n_iter = tk.n_iter
    n_iter_per_ss = tk.ss_n_iter
    n_iter_between_ss = tk.ss_n_iter_between

    hist_pr = \
        {
            "post_mean": np.zeros((n_obs, n_param)),
            "post_std": np.zeros((n_obs, n_param))
        }
    c_iter_ss = 0
    for begin_ss in np.arange(0, n_iter, n_iter_per_ss):

        for key in hist_pr.keys():
            split = \
                param_recovery_cond[key][begin_ss:begin_ss + n_iter_per_ss]
            last = split[-1, :]
            t = c_iter_ss * (n_iter_per_ss + n_iter_between_ss)
            start, end = t, t + n_iter_per_ss
            hist_pr[key][start: end] = split
            start, end = end, end + n_iter_between_ss
            hist_pr[key][start: end] = last

        c_iter_ss += 1

    return hist_pr


def _format_data(data_cond, training, tk):
    n_obs = tk.terminal_t

    hist = data_cond

    seen_in_the_end = list(np.unique(hist))

    n_seen = np.zeros(n_obs, dtype=int)
    n_learnt = np.zeros(n_obs, dtype=int)
    p = [[] for _ in seen_in_the_end]

    it = 0

    psychologist = Psychologist.create(tk)

    now = 0

    for t in tqdm(range(tk.terminal_t)):

        t_is_teaching = training[t]

        p_at_t, seen = psychologist.p_seen(
            now=now,
            param=tk.param)

        sum_seen = np.sum(seen)

        n_seen[t] = sum_seen

        if sum_seen > 0:

            items_seen_at_t = np.arange(tk.n_item)[seen]

            for item, p_item in zip(items_seen_at_t, p_at_t):
                idx_in_the_end = seen_in_the_end.index(item)
                tup = (t, p_item)
                p[idx_in_the_end].append(tup)

            n_learnt[t] = np.sum(p_at_t > tk.learnt_threshold)

        if t_is_teaching:
            item = hist[it]
            psychologist.update_minimal(item=item, timestamp=now)
            it += 1
            now += tk.time_per_iter
        else:
            now += tk.time_per_iter * tk.ss_n_iter_between

    return {'n_learnt': n_learnt, 'p': p, 'n_seen': n_seen}


def make_fig(data, param_recovery, tk):

    # training = np.zeros(tk.terminal_t, dtype=bool)
    training = np.tile(
        [1, ]*tk.ss_n_iter
        + [0, ]*tk.ss_n_iter_between,
        tk.n_ss)

    cond_labels = list(data.keys())
    cond_labels_param_recovery = list(param_recovery.keys()) \
        if param_recovery is not None else []

    print(f"Cond for param recovery: {cond_labels_param_recovery}")

    data_fig = DataFig(cond_labels=cond_labels,
                       training=training,
                       info=tk.info,
                       threshold=tk.learnt_threshold,
                       exam=tk.terminal_t,
                       param=tk.param,
                       param_labels=tk.param_labels,
                       cond_labels_param_recovery=cond_labels_param_recovery)

    for i, cd in enumerate(cond_labels_param_recovery):
        # if len(tk.param.shape) > 1:  # Heterogeneous parameters
        hist_pr = {
            "post_mean": param_recovery[cd],
            "post_std": None
        }
        # else:
        #     hist_pr = _format_parameter_recovery(
        #         tk=tk,
        #         param_recovery_cond=param_recovery[cd])

        data_fig.add(
            post_mean=hist_pr["post_mean"],
            post_std=hist_pr["post_std"],
        )

    tqdm.write("Computing probabilities of recalls...")

    for i, cd in enumerate(cond_labels):

        formatted_data = _format_data(data_cond=data[cd],
                                      training=training, tk=tk)

        data_fig.add(
            n_learnt=formatted_data['n_learnt'],
            n_seen=formatted_data['n_seen'],
            p=formatted_data['p'],
        )

    plot(data=data_fig, fig_folder=FIG_FOLDER, fig_name=tk.extension)
