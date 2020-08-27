import numpy as np
import datetime
import pandas as pd

from model.teacher.leitner import Leitner
from model.teacher.threshold import Threshold
from model.teacher.sampling import Sampling

from model.learner.walsh2018 import Walsh2018
from model.learner.exponential_n_delta import ExponentialNDelta
# from model.learner.act_r2008 import ActR2008


def run(config):

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

    # n_iter = n_ss * ss_n_iter

    teacher_cls = config.teacher_cls
    psy_cls = config.psy_cls
    learner_cls = config.learner_cls

    bounds = config.bounds

    if teacher_cls == Leitner:
        teacher = teacher_cls.create(n_item=n_item, **teacher_pr)

    elif teacher_cls == Threshold:
        teacher = teacher_cls.create(n_item=n_item,
                                     learnt_threshold=learnt_threshold)
    elif teacher_cls == Sampling:
        teacher = teacher_cls.create(
            n_item=n_item, learnt_threshold=learnt_threshold,
            time_per_iter=time_per_iter,
            ss_n_iter=ss_n_iter, time_between_ss=time_between_ss,
            **teacher_pr)
    else:
        raise ValueError

    is_leitner = teacher_cls == Leitner
    is_sampling = teacher_cls == Sampling

    # if "horizon" in teacher_pr:
    #     horizon = teacher_pr["horizon"]
    # else:
    #     horizon = 0

    # if learner_cls in (ActR2008,):
    #     if is_item_specific:
    #         raise NotImplementedError
    #     else:
    #         learner = learner_cls(n_item=n_item,
    #                               n_iter=n_ss * ss_n_iter
    #                                      + horizon)
    if learner_cls in (Walsh2018, ExponentialNDelta):
        learner = learner_cls(n_item=n_item,
                              n_iter=n_ss * ss_n_iter,
                              cst_time=cst_time)
    else:
        raise ValueError

    if omniscient or is_leitner:
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
    ts = None
    was_success = None

    itr = 0

    # with tqdm(total=n_iter, file=sys.stdout) as pbar:
    for i in range(n_ss):
        for j in range(ss_n_iter):

            if item is not None and ts is not None:
                psy.update(item=item, response=was_success,
                           timestamp=ts)

                if is_leitner:
                    item = teacher.ask(now=now,
                                       last_was_success=was_success,
                                       last_time_reply=ts,
                                       idx_last_q=item)

                elif is_sampling:
                    item = teacher.ask(now=now,
                                       psy=psy,
                                       ss_iter=j)

                else:
                    item = teacher.ask(now=now,
                                       psy=psy)

                p = psy.p(
                    item=item,
                    param=pr,
                    now=ts)
            else:
                item = 0
                p = 0

            ts = now
            was_success = np.random.random() < p

            p_seen_real, seen = psy.p_seen(now=now, param=pr)

            n_learnt = np.sum(p_seen_real > learnt_threshold)
            # if j == 0:
            #     print("********n_learnt", n_learnt)
            # else:
            #     print("n_learnt", n_learnt)

            if not is_leitner or not omniscient:
                pr_inf = psy.inferred_learner_param()
                p_seen_inf, seen = psy.p_seen(now=now, param=pr_inf)

                if len(p_seen_inf):
                    p_err = np.abs(p_seen_real-p_seen_inf)
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

            # pbar.update()

        now += time_between_ss

    return pd.DataFrame(row_list)
