import numpy as np


def objective(model, tk, data, param, show=False):

    if show:
        print('\n')

    agent = model(param=param, tk=tk)
    t_max = data.t_max
    diff = np.zeros(t_max)

    for t in range(t_max):
        item = data.questions[t]
        if show:
            print("Item", item)
        p_r = agent.p_recall(item=item)
        if show:
            print('p_recall: ', p_r)
        s = data.success[t]
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
