import numpy as np
import multiprocessing as mlt


def run(args):

    agent, item, t_idx = args
    p = \
        agent.p_recall(item=item, time_index=t_idx)

    # assert not np.isnan(p), "Error of logic!"

    return p


def _compute_discrete(agent, n_item, t_max):

    p_recall = np.zeros((n_item, t_max))

    pool = mlt.Pool()

    for t_idx in range(t_max):

        p_recall[:, t_idx] = pool.map(
            run,
            ((agent, i, t_idx) for i in range(n_item)))

    return p_recall


def p_recall_over_time_after_learning(
        agent, t_max, n_item,
        discrete_time=True,
        time_norm=None,
        time_sampling=None):

    if time_sampling is not None:
        discrete_time = False

    assert discrete_time or \
        (time_sampling is not None and time_norm is not None), \
        "If 'discrete_time' is True, then " \
        "'time_norm' and 'time_sampling' have to be defined"

    if discrete_time:
        p_recall = _compute_discrete(agent=agent, n_item=n_item, t_max=t_max)

    else:

        samp_size = time_sampling.shape[0]

        p_recall = np.zeros((n_item, samp_size))

        for t_idx, t in enumerate(time_sampling):

            for item in range(n_item):
                p = \
                    agent.p_recall(item=item, time=t * time_norm)

                # assert not np.isnan(p), "Error of logic!"

                p_recall[item, t_idx] = p

    return p_recall
