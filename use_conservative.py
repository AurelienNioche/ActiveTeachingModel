import numpy as np


def _threshold_select(n_pres, param, n_item, is_item_specific,
                      ts, last_pres, cst_time, thr):

    log_thr = np.log(thr)
    if np.max(n_pres) == 0:
        item = 0
    else:
        seen = n_pres > 0

        log_p_seen = _cp_log_p_seen(
            seen=seen,
            n_pres=n_pres,
            param=param,
            n_item=n_item,
            is_item_specific=is_item_specific,
            last_pres=last_pres,
            ts=ts,
            cst_time=cst_time)

        if np.sum(seen) == n_item or np.min(log_p_seen) <= log_thr:
            item = np.flatnonzero(seen)[np.argmin(log_p_seen)]
        else:
            item = np.argmin(seen)

    return item


def _cp_log_p_seen(seen, n_pres, param, n_item, is_item_specific,
                   last_pres, ts, cst_time):
    if is_item_specific:
        init_forget = param[:n_item][seen, 0]
        rep_effect = param[:n_item][seen, 1]
    else:
        init_forget, rep_effect = param

    return \
        -init_forget \
        * (1 - rep_effect) ** (n_pres[seen] - 1) \
        * (ts - last_pres[seen]) \
        * cst_time


def _select(n_pres, last_pres,
            future_ts, param, eval_ts,
            cst_time, is_item_specific, n_item,
            thr):

    now = future_ts[0]
    future = future_ts[1:]

    n_pres_current = n_pres
    last_pres_current = last_pres

    while True:

        n_pres = n_pres_current[:n_item]
        last_pres = last_pres_current[:n_item]

        first_item = _threshold_select(
            n_pres=n_pres,
            param=param,
            n_item=n_item,
            is_item_specific=is_item_specific,
            ts=now, last_pres=last_pres,
            cst_time=cst_time,
            thr=thr)

        n_item = first_item + 1

        n_pres = n_pres_current[:n_item].copy()
        last_pres = last_pres_current[:n_item].copy()

        n_pres[first_item] += 1
        last_pres[first_item] = now

        for ts in future:

            item = _threshold_select(
                n_pres=n_pres,
                param=param,
                n_item=n_item,
                is_item_specific=is_item_specific,
                ts=ts, last_pres=last_pres,
                cst_time=cst_time,
                thr=thr)

            n_pres[item] += 1
            last_pres[item] = ts

        seen = n_pres > 0
        log_p_seen = _cp_log_p_seen(
            seen=seen,
            n_pres=n_pres,
            param=param,
            n_item=n_item,
            is_item_specific=is_item_specific,
            last_pres=last_pres,
            ts=eval_ts,
            cst_time=cst_time)

        n_learnt = np.sum(log_p_seen > np.log(thr))
        if n_learnt == n_item:
            break

        n_item = first_item
        if n_item <= 1:
            break

    return first_item


def use_conservative(n_item, param, review_ts, eval_ts, thr):

    is_item_specific = len(np.asarray(param).shape) > 1

    n_pres = np.zeros(n_item, dtype=int)
    last_pres = np.zeros(n_item)

    for i, ts in enumerate(review_ts):

        if np.max(n_pres) == 0:
            item = 0
        else:

            current_step = i
            future_ts = review_ts[current_step:]

            item = _select(
                is_item_specific=is_item_specific,
                future_ts=future_ts,
                cst_time=1,
                eval_ts=eval_ts,
                param=param,
                n_pres=n_pres,
                last_pres=last_pres,
                n_item=n_item,
                thr=thr)

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
    print("conservative n learnt", n_learnt, "n seen", np.sum(seen))
    print()


def main():

    n_item = 150

    param = [0.0006, 0.44]  # [[0.0006, 0.44] for _ in range(n_item)]
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

    use_conservative(n_item, param, review_ts, eval_ts, thr)


if __name__ == "__main__":
    main()
