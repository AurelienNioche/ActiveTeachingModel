import numpy as np


def objective(model,
              t,
              hist_item, hist_success,
              task_param,
              param):

    current_n_iteration = t+1
    agent = model(param=param, n_iteration=current_n_iteration,
                  **task_param)
    diff = np.zeros(current_n_iteration)

    for t in range(current_n_iteration):
        item = hist_item[t]
        p_r = agent.p(item=item)
        s = hist_success[t]

        diff[t] = (s - p_r)

        agent.learn(item)

    # diff[diff < 0] *= 1.1
    diff = np.power(diff, 2)
    value = np.mean(diff)
    assert value >= 0
    return value
