import numpy as np

from adaptive_teaching.simplified.learner import learner_fr_p_seen


def get_item(n_pres, n_success, param, delta):

    n_item = len(n_pres)
    seen = n_pres[:] > 0
    unseen = np.logical_not(seen)
    n_seen = np.sum(seen)

    items = np.arange(n_item)

    if n_seen == 0:
        return np.random.randint(n_item)

    fr_seen, pr_seen = learner_fr_p_seen(n_success, param, n_pres, delta)
    min_pr_seen = np.min(pr_seen)
    if min_pr_seen < 0.90 or n_seen == n_item:
        return np.random.choice(items[seen][pr_seen[:] == min_pr_seen])
    else:
        return np.random.choice(items[unseen])
