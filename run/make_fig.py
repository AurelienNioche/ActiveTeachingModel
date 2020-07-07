import os
import sys

import numpy as np
from tqdm import tqdm

from teacher.psychologist.psychologist import Psychologist

from plot.plot import DataFig, plot

SCRIPT_NAME = os.path.basename(__file__).split(".")[0]

FIG_FOLDER = os.path.join("fig", SCRIPT_NAME)
os.makedirs(FIG_FOLDER, exist_ok=True)


# def _format_parameter_recovery(param_recovery_cond, tk):
#
#     n_obs = tk.terminal_t
#     n_param = len(tk.param)
#
#     n_iter = tk.n_iter
#     n_iter_per_ss = tk.ss_n_iter
#     n_iter_between_ss = tk.ss_n_iter_between
#
#     hist_pr = \
#         {
#             "post_mean": np.zeros((n_obs, n_param)),
#             "post_std": np.zeros((n_obs, n_param))
#         }
#     c_iter_ss = 0
#     for begin_ss in np.arange(0, n_iter, n_iter_per_ss):
#
#         for key in hist_pr.keys():
#             split = \
#                 param_recovery_cond[key][begin_ss:begin_ss + n_iter_per_ss]
#             last = split[-1, :]
#             t = c_iter_ss * (n_iter_per_ss + n_iter_between_ss)
#             start, end = t, t + n_iter_per_ss
#             hist_pr[key][start: end] = split
#             start, end = end, end + n_iter_between_ss
#             hist_pr[key][start: end] = last
#
#         c_iter_ss += 1
#
#     return hist_pr


def _format_data(data_cond, training, tk, limit_dsp_btn):

    if tk.ss_n_iter_between > limit_dsp_btn:
        clip_size = tk.ss_n_iter_between - limit_dsp_btn
    else:
        clip_size = None

    c_tch = 0        # Counter teaching iteration
    c_t = 0          # Counter time step
    c_slice_btn = 0  # Counter slice between
    now = 0          # 'now' (sec)

    n_obs = tk.n_ss * (tk.ss_n_iter + min(tk.ss_n_iter_between,
                                          limit_dsp_btn))

    hist = data_cond
    seen_in_the_end = list(np.unique(hist))

    n_seen = np.zeros(n_obs, dtype=int)
    n_learnt = np.zeros(n_obs, dtype=int)
    timestamps = np.zeros(n_obs, dtype=int)
    p = [[] for _ in seen_in_the_end]

    psychologist = Psychologist.create(tk=tk, omniscient=True)

    for c_obs in tqdm(range(n_obs), file=sys.stdout):

        jump = 1
        t_is_teaching = training[c_t]

        if c_obs == 0:
            n_seen[c_obs] = 0

        else:
            p_at_t, seen = psychologist.p_seen(now=now)
            n_seen[c_obs] = np.sum(seen)

            items_seen_at_t = np.flatnonzero(seen)

            for item, p_item in zip(items_seen_at_t, p_at_t):
                idx_in_the_end = seen_in_the_end.index(item)
                tup = (now, p_item)
                p[idx_in_the_end].append(tup)

            n_learnt[c_obs] = np.sum(p_at_t > tk.learnt_threshold)

        if t_is_teaching:
            item = hist[c_tch]
            psychologist.update_learner(item=item, timestamp=now)
            c_tch += 1

        elif clip_size is not None:
            c_slice_btn += 1
            if c_slice_btn == limit_dsp_btn/2:
                jump += clip_size
            elif c_slice_btn == limit_dsp_btn:
                c_slice_btn = 0

        timestamps[c_obs] = now
        c_t += jump
        now += jump * tk.time_per_iter

    return {'n_learnt': n_learnt, 'p': p, 'n_seen': n_seen,
            'timestamps': timestamps}


def make_fig(data, param_recovery, tk, limit_dsp_btn=200):

    assert limit_dsp_btn % 2 == 0

    # training = np.zeros(tk.terminal_t, dtype=bool)
    training = np.tile(
        [1, ]*tk.ss_n_iter
        + [0, ]*tk.ss_n_iter_between,
        tk.n_ss)

    exam = tk.n_ss * (tk.ss_n_iter_between + tk.ss_n_iter) * tk.time_per_iter

    if limit_dsp_btn is not None and tk.ss_n_iter_between > limit_dsp_btn:
        xlims = np.zeros((tk.n_ss+1, 2), dtype=int)
        xlims[:, 0] = np.arange(tk.n_ss+1) * (tk.ss_n_iter_between+tk.ss_n_iter) - limit_dsp_btn/2
        xlims[:, 1] = xlims[:, 0] + tk.ss_n_iter + limit_dsp_btn
        xlims[0, :] = 0, tk.ss_n_iter + limit_dsp_btn/2
        xlims[-1, 1] = tk.n_ss * (tk.ss_n_iter_between + tk.ss_n_iter)
        xlims *= tk.time_per_iter

    else:
        xlims = None

    cond_labels = list(data.keys())
    cond_labels_param_recovery = list(param_recovery.keys()) \
        if param_recovery is not None else []

    print(f"Cond for param recovery: {cond_labels_param_recovery}")

    data_fig = DataFig(
        xlims=xlims,
        cond_labels=cond_labels,
        time_per_iter=tk.time_per_iter,
        training=training,
        info=tk.info,
        threshold=tk.learnt_threshold,
        exam=exam,
        param=tk.param,
        param_labels=tk.param_labels,
        cond_labels_param_recovery=cond_labels_param_recovery)

    for i, cd in enumerate(cond_labels_param_recovery):
        pr = param_recovery[cd]
        tr = np.flatnonzero(training)
        if training[-1] == 0:
            tr = np.hstack((tr, len(training)-1))
            pr = np.vstack((pr, np.tile(pr[-1], 1)))
        tr *= tk.time_per_iter
        data_fig.add(
            timestamps_recovery=tr,
            inferred_param=pr)

    tqdm.write("Computing probabilities of recalls...")

    for i, cd in enumerate(cond_labels):

        data_fig.add(**_format_data(data_cond=data[cd],
                                    training=training, tk=tk,
                                    limit_dsp_btn=limit_dsp_btn))

    plot(data=data_fig, fig_folder=FIG_FOLDER, fig_name=tk.extension)
