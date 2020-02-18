import numpy as np


class Teacher:

    def __init__(self):
        pass

    @staticmethod
    def get_item(learner, n_pres, n_success, param, delta, hist,
                 timestamps, t):

        n_item = len(n_pres)
        seen = n_pres[:] > 0
        unseen = np.logical_not(seen)
        n_seen = np.sum(seen)

        items = np.arange(n_item)

        if n_seen == 0:
            return np.random.randint(n_item)

        pr_seen = learner.p_seen(n_success=n_success,
                                 param=param,
                                 n_pres=n_pres,
                                 delta=delta, hist=hist,
                                 timestamps=timestamps, t=t)
        min_pr_seen = np.min(pr_seen)

        # if min_pr_seen <= 0.90:
        #     max_under = np.max(pr_seen[pr_seen <= 90])
        #     return np.random.choice(items[seen][pr_seen[:] == max_under])

        if n_seen == n_item or min_pr_seen <= 0.90:
            return np.random.choice(items[seen][pr_seen[:] == min_pr_seen])

        else:
            return np.random.choice(items[unseen])
