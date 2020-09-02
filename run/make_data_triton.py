import numpy as np
import datetime
import pandas as pd
from tqdm import tqdm

from model.teacher.leitner import Leitner
from model.teacher.threshold import Threshold
from model.teacher.sampling import Sampling
from model.teacher.recursive import Recursive

from model.learner.walsh2018 import Walsh2018
from model.learner.exponential_n_delta import ExponentialNDelta
# from model.learner.act_r2008 import ActR2008


def run(config, with_tqdm=False):

    n_item = config.n_item
    omniscient = config.omniscient
    config_file = config.config_file
    config_dic = config.config_dic
    seed = config.seed
    n_ss = config.n_ss
    ss_n_iter = config.ss_n_iter
    time_between_ss = config.time_between_ss
    time_per_iter = config.time_per_iter
    is_item_specific = config.is_item_specific
    learnt_threshold = config.learnt_threshold
    cst_time = config.cst_time

    pr = config.param
    teacher_pr = config.teacher_pr
    psy_pr = config.psy_pr

    teacher_cls = config.teacher_cls
    psy_cls = config.psy_cls
    learner_cls = config.learner_cls

    bounds = config.bounds

    if teacher_cls == Leitner:
        teacher = teacher_cls(n_item=n_item, **teacher_pr)

    elif teacher_cls == Threshold:
        teacher = teacher_cls(n_item=n_item,
                              learnt_threshold=learnt_threshold)
    elif teacher_cls == Sampling:
        teacher = teacher_cls(
            n_item=n_item, learnt_threshold=learnt_threshold,
            time_per_iter=time_per_iter,
            ss_n_iter=ss_n_iter, time_between_ss=time_between_ss,
            **teacher_pr)
    elif teacher_cls == Recursive:
        teacher = teacher_cls(n_item=n_item,
                              learnt_threshold=learnt_threshold,
                              time_per_iter=time_per_iter,
                              n_ss=n_ss,
                              ss_n_iter=ss_n_iter,
                              time_between_ss=time_between_ss)

    else:
        raise ValueError

    is_leitner = teacher_cls == Leitner
    is_sampling = teacher_cls == Sampling

    if learner_cls in (Walsh2018, ExponentialNDelta):
        learner = learner_cls(n_item=n_item,
                              n_iter=n_ss * ss_n_iter)
    else:
        raise ValueError

    if omniscient or is_leitner:
        psy = psy_cls(
            n_item=n_item,
            is_item_specific=is_item_specific,
            learner=learner,
            bounds=bounds,
            cst_time=cst_time,
            true_param=pr,
            **psy_pr)
    else:
        psy = psy_cls(
            n_item=n_item,
            is_item_specific=is_item_specific,
            learner=learner,
            bounds=bounds,
            cst_time=cst_time,
            **psy_pr)

    delta_end_ss_begin_ss = time_between_ss - time_per_iter * ss_n_iter

    np.random.seed(seed)

    row_list = []

    now = 0.0

    item = None
    ts = None
    was_success = None

    itr = 0

    if with_tqdm:
        import sys
        n_iter = n_ss * ss_n_iter
        pbar = tqdm(total=n_iter, file=sys.stdout)

    for i in range(n_ss):
        for j in range(ss_n_iter):

            if item is None and ts is None:
                item = 0
                p = 0
                n_learnt_before = 0
                n_seen_before = 0
            else:

                p_seen_real_before, seen_before = psy.p_seen(now=now, param=pr)
                n_learnt_before = np.sum(p_seen_real_before > learnt_threshold)
                n_seen_before = np.sum(seen_before)

                if is_leitner:
                    item = teacher.ask(now=now,
                                       last_was_success=was_success,
                                       last_time_reply=ts,
                                       idx_last_q=item)

                elif is_sampling:
                    item = teacher.ask(now=now, psy=psy, ss_iter=j)

                else:
                    item = teacher.ask(now=now, psy=psy)

                p = psy.p(item=item, param=pr, now=ts)

            ts = now
            was_success = np.random.random() < p

            psy.update(item=item, response=was_success, timestamp=ts)

            p_seen_real, seen = psy.p_seen(now=now, param=pr)

            n_learnt = np.sum(p_seen_real > learnt_threshold)
            n_seen = np.sum(seen)
            # if j == 0:
            #     print("********n_learnt", n_learnt)
            # else:
            #     print("n_learnt", n_learnt)

            if is_leitner or omniscient:
                pr_inf, p_err_mean, p_err_std = None, None, None

            else:
                pr_inf = psy.inferred_learner_param()
                p_seen_inf, seen = psy.p_seen(now=now, param=pr_inf)

                if np.sum(seen):
                    p_err = np.abs(p_seen_real - p_seen_inf)
                    p_err_mean, p_err_std = np.mean(p_err), np.std(p_err)
                else:
                    p_err_mean, p_err_std = None, None

            now_real = datetime.datetime.now().timestamp()

            row = {
                "iter": itr,
                "item": item,
                "success": was_success,
                "ss_idx": i,
                "ss_iter": j,
                "pr_inf": pr_inf,
                "p_err_mean": p_err_mean,
                "p_err_std": p_err_std,
                "n_learnt_before": n_learnt_before,
                "n_learnt": n_learnt,
                "n_seen_before": n_seen_before,
                "n_seen": n_seen,
                "timestamp": now,
                "timestamp_cpt": now_real,
                "config_file": config_file,
                "is_human": False,
                **config_dic
            }
            row_list.append(row)

            now += time_per_iter
            itr += 1

            if with_tqdm:
                pbar.update()

        now += delta_end_ss_begin_ss

    p_seen_real, seen = psy.p_seen(now=now, param=pr)

    n_learnt = np.sum(p_seen_real > learnt_threshold)
    n_seen = np.sum(seen)

    if with_tqdm:
        pbar.close()
        print()
        print("now", now, "n_learnt", n_learnt, "n_seen", n_seen)

    row = {
        "iter": itr,
        "item": item,
        "success": was_success,
        "ss_idx": n_ss,
        "ss_iter": ss_n_iter,
        "pr_inf": None,
        "p_err_mean": None,
        "p_err_std": None,
        "n_learnt_before": None,
        "n_learnt": n_learnt,
        "n_seen_before": None,
        "n_seen": n_seen,
        "timestamp": now,
        "timestamp_cpt": datetime.datetime.now().timestamp(),
        "config_file": config_file,
        "is_human": False,
        **config_dic
    }
    row_list.append(row)

    return pd.DataFrame(row_list)
