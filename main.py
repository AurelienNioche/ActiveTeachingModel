import numpy as np

def p_recall(item, is_item_specific, param, n_pres, last_pres, ts):

    if is_item_specific:
        init_forget = param[item, 0]
        rep_effect = param[item, 1]
    else:
        init_forget, rep_effect = param

    return 0 if not n_pres[item] \
        else np.exp(
        -init_forget
        * (1 - rep_effect) ** (n_pres[item] - 1)
        * (ts - last_pres[item]))


def log_p_seen(n_pres, is_item_specific, param, eval_ts, last_pres):

    seen = n_pres > 0

    if is_item_specific:
        init_forget = param[seen, 0]
        rep_effect = param[seen, 1]
    else:
        init_forget, rep_effect = param

    return np.exp(
        -init_forget
        * (1 - rep_effect) ** (n_pres[seen] - 1)
        * (eval_ts - last_pres[seen]))


def all_p(n_pres, is_item_specific, param, eval_ts, last_pres):

    seen = n_pres > 0

    if is_item_specific:
        init_forget = param[seen, 0]
        rep_effect = param[seen, 1]
    else:
        init_forget, rep_effect = param

    fr = -init_forget * (1 - rep_effect) ** (n_pres[seen] - 1)
    _p_seen = np.exp(fr * (eval_ts - last_pres[seen]))

    _all_p = np.zeros(len(seen))
    _all_p[seen] = _p_seen

    return _all_p


def run(n_item, review_ts, param, eval_ts, thr):

    is_item_specific = len(np.asarray(param).shape) > 1

    n_pres = np.zeros(n_item, dtype=int)
    last_pres = np.zeros(n_item)

    for i, ts in enumerate(review_ts):

        item =

        p = p_recall(item=item, is_item_specific=is_item_specific,
                     n_pres=n_pres, last_pres=last_pres, ts=ts,
                     param=param)

        n_pres[item] += 1
        last_pres[item] = ts

    seen = n_pres > 0

    if is_item_specific:
        init_forget = param[seen, 0]
        rep_effect = param[seen, 1]
    else:
        init_forget, rep_effect = param

    p_seen = np.exp(
        -init_forget
        * (1 - rep_effect) ** (n_pres[seen] - 1)
        * (eval_ts - last_pres[seen]))

    n_learnt = np.sum(p_seen > thr)
    print("n learnt", n_learnt)
    print()


def main():

    n_item = 150

    param = [[0.0006, 0.44] for _ in range(n_item)]
    param = np.asarray(param)

    ss_n_iter = 100
    time_per_iter = 2
    n_sec_day = 24 * 60 ** 2
    n_ss = 6
    eval_ts = n_ss * n_sec_day
    review_ts = np.hstack(
        [
            np.arange(x, x + (ss_n_iter * time_per_iter), time_per_iter)
            for x in np.arange(0, n_sec_day * n_ss, n_sec_day)
        ]
    )

    thr = 0.90

    n_total_iter = len(review_ts)

    np.random.seed(123)






if __name__ == "__main__":
    main()
