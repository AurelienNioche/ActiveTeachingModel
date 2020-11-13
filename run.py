import datetime

import numpy as np
from tqdm import tqdm

from exploration_exploitation import backward_induction


def run(n_item, param, review_ts, eval_ts, thr):

    n_pres = np.zeros(n_item, dtype=int)
    last_pres = np.zeros(n_item)

    for i, ts in enumerate(review_ts):

        if np.max(n_pres) == 0:
            item = 0
        else:
            u = backward_induction.run(
                review_ts=review_ts,
                param=param,
                thr=thr,
                eval_ts=eval_ts,
                n_pres_current=n_pres,
                idx_ts_current=i)

            item = np.argmax(u)

        n_pres[item] += 1
        last_pres[item] = ts

    seen = n_pres > 0

    init_forget, rep_effect = param

    log_p_seen = \
        -init_forget \
        * (1 - rep_effect) ** (n_pres[seen] - 1) \
        * (eval_ts - last_pres[seen])

    n_learnt = np.sum(log_p_seen > np.log(thr))
    print("n learnt", n_learnt)
    print()


def main():

    np.random.seed(123)

    n_item = 100

    param = [0.0006, 0.44]
    param = np.asarray(param)

    ss_n_iter = 100
    time_per_iter = 4
    n_sec_day = 24 * 60 ** 2
    n_ss = 6
    eval_ts = n_ss * n_sec_day
    review_ts = np.hstack([
            np.arange(x, x + (ss_n_iter * time_per_iter), time_per_iter)
            for x in np.arange(0, n_sec_day * n_ss, n_sec_day)
        ])

    thr = 0.90

    a = datetime.datetime.now()

    run(
        n_item=n_item, param=param, review_ts=review_ts,
        eval_ts=eval_ts, thr=thr)

    print("[Time to execute ", datetime.datetime.now() - a, "]")


if __name__ == "__main__":
    main()