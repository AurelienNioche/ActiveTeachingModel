import numpy as np
import datetime

from use_threshold import use_threshold
from use_leitner import use_leitner
from use_conservative import use_conservative


def main():

    n_item = 50

    param = [0.006, 0.44]    # [[0.001, 0.44] for _ in range(n_item)]
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

    np.random.seed(123)

    a = datetime.datetime.now()

    use_leitner(
        n_item=n_item, param=param, review_ts=review_ts,
        eval_ts=eval_ts, thr=thr)

    use_threshold(
        n_item=n_item, param=param, review_ts=review_ts,
        eval_ts=eval_ts, thr=thr, eval_thr=thr)

    use_conservative(
        n_item=n_item,
        param=param,
        review_ts=review_ts,
        eval_ts=eval_ts,
        thr=thr)

    print("[Time to execute ", datetime.datetime.now() - a, "]")


if __name__ == "__main__":
    main()
