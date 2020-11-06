import numpy as np
import datetime


def run(review_ts, param, thr, eval_ts):

    alpha, beta = param

    empty = np.ones(len(review_ts), dtype=bool)

    n_review_max = len(review_ts)

    n_item = 0
    n_pres = 1
    n_review = 0
    while True:
        max_delay = - np.log(thr) / (alpha * (1 - beta) ** (n_pres - 1))
        # print("n item", n_item)
        # print("n pres", n_pres)
        # print("max delay", max_delay)
        min_ts = eval_ts - max_delay
        # print("min ts", min_ts)
        bool_possible = review_ts[empty] > min_ts
        idx_possible = np.flatnonzero(bool_possible)
        # print("idx possible", idx_possible)
        is_place = len(idx_possible)
        if is_place:
            potential_idx = idx_possible[-1]
            is_possible = n_review + n_pres <= n_review_max
            # print("n review", n_review)
            # print("is possible", is_possible)
            if is_possible:
                empty[potential_idx] = False
                full = np.sum(empty) == 0
                print(n_pres)
                n_review += n_pres
                n_item += 1
                if full:
                    break
            else:
                break
        else:
            n_pres += 1
        # print()

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
