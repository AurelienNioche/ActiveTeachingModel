from tqdm import tqdm
import numpy as np
import sys
import datetime
import pandas as pd

from model.teacher.leitner import Leitner

from model.learner.act_r2008 import ActR2008
from model.learner.walsh2018 import Walsh2018
from model.learner.exponential_n_delta import ExponentialNDelta


def run(config):

    n_item = config.n_item
    omniscient = config.omniscient
    config_file = config.config_file
    config_dic = config.config_dic
    seed = config.seed

    pr = config.param
    teacher_pr = config.teacher_pr
    task_pr = config.task_pr
    psy_pr = config.psy_pr

    n_ss = task_pr.n_ss
    ss_n_iter = task_pr.ss_n_iter
    time_between_ss = task_pr.time_between_ss
    time_per_iter = task_pr.time_per_iter
    is_item_specific = task_pr.is_item_specific
    learnt_threshold = task_pr.learnt_threshold

    n_iter = n_ss * ss_n_iter

    teacher_cls = config.teacher_cls
    psy_cls = config.psy_cls
    learner_cls = config.learner_cls

    bounds = config.bounds

    teacher = teacher_cls.create(task_pr=task_pr,
                                 n_item=n_item, **teacher_pr)
    teacher_use_psy = teacher_cls != Leitner

    if "horizon" in teacher_pr:
        horizon = teacher_pr["horizon"]
    else:
        horizon = 0

    if learner_cls in (ActR2008,):
        if task_pr.is_item_specific:
            raise NotImplementedError
        else:
            learner = learner_cls(n_item=n_item,
                                  n_iter=n_ss * ss_n_iter
                                         + horizon)
    elif learner_cls in (Walsh2018, ExponentialNDelta):
        learner = learner_cls(n_item=n_item,
                              n_iter=n_ss * ss_n_iter
                                     + horizon)
    else:
        raise ValueError

    if omniscient or not teacher_use_psy:
        psy = psy_cls.create(
            n_item=n_item,
            is_item_specific=is_item_specific,
            learner=learner,
            bounds=bounds,
            **psy_pr, true_param=pr)
    else:
        psy = psy_cls.create(
            n_item=n_item,
            is_item_specific=is_item_specific,
            learner=learner,
            bounds=bounds,
            **psy_pr,)

    np.random.seed(seed)

    row_list = []

    now = 0

    item = None
    timestamp = None
    was_success = None

    itr = 0

    with tqdm(total=n_iter, file=sys.stdout) as pbar:
        for i in range(n_ss):
            for j in range(ss_n_iter):

                if item is not None and timestamp is not None:
                    psy.update(item=item, response=was_success,
                               timestamp=timestamp)

                    if teacher_use_psy:
                        item = teacher.ask(now=now,
                                           psy=psy)
                    else:
                        item = teacher.ask(now=now,
                                           last_was_success=was_success,
                                           last_time_reply=timestamp,
                                           idx_last_q=item)
                    p = psy.p(
                        item=item,
                        param=pr,
                        now=timestamp)
                else:
                    item = 0
                    p = 0

                timestamp = now
                was_success = np.random.random() < p

                real_p_seen, seen = \
                    psy.p_seen(now=now, param=pr)

                n_learnt = np.sum(real_p_seen > learnt_threshold)

                if teacher_use_psy:
                    pr_inf = psy.inferred_learner_param()
                    inferred_p_seen, seen = \
                        psy.p_seen(now=now, param=pr_inf)

                    if len(inferred_p_seen):
                        p_err = np.abs(real_p_seen-inferred_p_seen)
                        p_err_mean, p_err_std = np.mean(p_err), np.std(p_err)
                    else:
                        p_err_mean, p_err_std = None, None

                else:
                    pr_inf, p_err_mean, p_err_std = None, None, None

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
                    "n_learnt": n_learnt,
                    "timestamp": now,
                    "timestamp_cpt": now_real,
                    "config_file": config_file,
                    "is_human": False
                }
                row.update(config_dic)
                row_list.append(row)

                now += time_per_iter
                itr += 1

                pbar.update()

            now += time_between_ss

    return pd.DataFrame(row_list)
