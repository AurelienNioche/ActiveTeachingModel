import numpy as np
import datetime
import itertools

def run(n_item, sequence, review_ts, is_item_specific, param,
        eval_ts, log_thr):

    n_pres = np.zeros(n_item, dtype=int)
    last_pres = np.zeros(n_item, dtype=int)

    current_seen_item = np.unique(sequence)
    for i, item in enumerate(current_seen_item):
        is_item = sequence == item
        n_pres[item] = np.sum(is_item)
        last_pres[item] = review_ts[is_item][-1]

    seen = n_pres > 0

    if is_item_specific:
        init_forget = param[seen, 0]
        rep_effect = param[seen, 1]
    else:
        init_forget, rep_effect = param

    log_p_seen = \
        -init_forget \
        * (1 - rep_effect) ** (n_pres[seen] - 1) \
        * (eval_ts - last_pres[seen])

    n_learnt = np.sum(log_p_seen > log_thr)
    return n_learnt


def use_progressive(n_item, review_ts, param, eval_ts, thr,
                    n_sample=10**5):

    log_thr = np.log(thr)
    is_item_specific = len(np.asarray(param).shape) > 1

    horizon = len(review_ts)

    test_n_item = 0
    while True:
        test_n_item += 1

        success = False
        i = 0

        n_perm = test_n_item ** horizon

        items = np.arange(test_n_item)

        use_perm = n_perm <= n_sample
        if use_perm:
            gen = itertools.product(items, repeat=horizon)
        else:
            gen = None

        while True:

            if use_perm:
                seq = next(gen)

            else:
                seq = np.random.choice(
                    items,
                    replace=True, size=horizon)

            n_learnt = run(
                n_item=test_n_item,
                sequence=seq, review_ts=review_ts,
                is_item_specific=is_item_specific,
                param=param, eval_ts=eval_ts, log_thr=log_thr)

            if n_learnt == test_n_item:
                print("n item", test_n_item, "n learnt", n_learnt, "i", i, "n perm", n_perm)
                success = True
                break

            else:
                i += 1
                if use_perm:
                    if i >= n_perm:
                        print("n item", test_n_item, "every perm explored", "i", i)
                        break
                elif i >= n_sample:
                    print("n item", test_n_item, "i", i)
                    break

        if success is False:
            n_learnt = np.max([n_learnt, test_n_item - 1])
            break
        elif n_learnt == n_item:
            n_learnt = test_n_item
            break
    print("n learnt", n_learnt)


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

    use_progressive(n_item=n_item, review_ts=review_ts, param=param,
                    eval_ts=eval_ts, thr=thr)
    print("[Time to execute ", datetime.datetime.now() - a, "]")


if __name__ == "__main__":
    main()
