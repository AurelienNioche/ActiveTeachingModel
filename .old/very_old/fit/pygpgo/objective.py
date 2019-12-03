import numpy as np


def objective(model,
              t,
              hist_item, hist_success,
              task_param,
              param, show=False):

    if show:
        print('\n')

    current_n_iteration = t+1
    agent = model(param=param, n_iteration=current_n_iteration,
                  **task_param)
    diff = np.zeros(current_n_iteration)

    for t in range(current_n_iteration):
        item = hist_item[t]
        if show:
            print("Item", item)
        p_r = agent.p(item=item)
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
