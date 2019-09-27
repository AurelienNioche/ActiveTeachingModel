import numpy as np
from tqdm import tqdm


def _compute_discrete(agent, n_item, n_iteration):

    p_recall = np.zeros((n_item, n_iteration))

    # pool = mlt.Pool()

    for t in tqdm(range(n_iteration)):

        for item in range(n_item):
            p = \
                agent.p_recall(item=item, time_index=t)

            # assert not np.isnan(p), "Error of logic!"

            p_recall[item, t] = p

        # p_recall[:, i] = pool.map(
        #     _run,
        #     ((agent, item, i) for item in range(n_item)))

    tqdm.write('\n')

    return p_recall


def _compute_continuous(agent, n_item, time_sampling,
                        time_norm):

    samp_size = time_sampling.shape[0]

    p_recall = np.zeros((n_item, samp_size))

    i = 0

    for t in tqdm(time_sampling):

        for item in range(n_item):
            p = \
                agent.p_recall(item=item, time=t * time_norm)

            # assert not np.isnan(p), "Error of logic!"

            p_recall[item, i] = p

        i += 1

    tqdm.write('\n')

    return p_recall


def p_recall_over_time_after_learning(
        agent, n_iteration, n_item,
        discrete_time=True,
        time_norm=None,
        time_sampling=None):

    tqdm.write("Computing the probabilities of recall...")

    if time_sampling is not None:
        discrete_time = False

    assert discrete_time or \
        (time_sampling is not None and time_norm is not None), \
        "If 'discrete_time' is True, then " \
        "'time_norm' and 'time_sampling' have to be defined"

    if discrete_time:
        p_recall = _compute_discrete(agent=agent, n_item=n_item, n_iteration=n_iteration)

    else:
        p_recall = _compute_continuous(agent=agent, n_item=n_item,
                                       time_sampling=time_sampling,
                                       time_norm=time_norm)

    return p_recall