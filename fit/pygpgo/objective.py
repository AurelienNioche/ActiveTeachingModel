import numpy as np


def objective(model, hist_item, hist_success,
              n_item, n_iteration,
              param, show=False):

    if show:
        print('\n')

    agent = model(param=param, n_iteration=n_iteration, n_item=n_item)
    diff = np.zeros(n_iteration)

    for t in range(n_iteration):
        item = hist_item[t]
        if show:
            print("Item", item)
        p_r = agent.p_recall(item=item)
        if show:
            print('p_recall: ', p_r)
        s = hist_success[t]
        if show:
            print("success: ", s)

        diff[t] = (s - p_r)

        agent.learn(item)

    # diff[diff < 0] *= 1.1
    diff = np.power(diff, 2)
    value = -np.sum(diff)
    if show:
        print("total value", value)
        print()
    return value
