import numpy as np
import datetime
import math


def run(review_ts, param, thr, eval_ts):

    alpha, beta = param

    n_review_max = len(review_ts)

    n_item = 0
    n_review = 0

    for ts in review_ts[::-1]:

        delta = eval_ts - ts
        min_n = (math.log(-(np.log(thr))) - math.log(alpha*delta)) / math.log(1 - beta) + 1
        n_pres = math.ceil(min_n)
        n_review += n_pres
        if n_review > n_review_max:
            break

        n_item += 1

    print("n learnt", n_item)


def main():

    np.random.seed(123)

    n_item = 50

    param = [0.006, 0.44]   # [[0.001, 0.44] for _ in range(n_item)]
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

    run(review_ts=review_ts, param=param, thr=thr, eval_ts=eval_ts)
    print("[Time to execute ", datetime.datetime.now() - a, "]")


if __name__ == "__main__":
    main()
