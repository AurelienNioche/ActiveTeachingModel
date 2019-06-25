import numpy as np


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

    if not discrete_time:

        samp_size = time_sampling.shape[0]

        p_recall = np.zeros((n_item, samp_size))

        for t_idx, t in enumerate(time_sampling):

            for item in range(n_item):
                p_recall[item, t_idx] = \
                    agent.p_recall(item=item, time=t*time_norm)

    else:
        p_recall = np.zeros((n_item, t_max))

        for t_idx in range(t_max):

            for item in range(n_item):
                p_recall[item, t_idx] = \
                    agent.p_recall(item=item, time_index=t_idx)

    return p_recall
