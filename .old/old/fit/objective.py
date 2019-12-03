import numpy as np


def objective(model,
              hist_question,
              hist_success,
              param,
              task_param,):

    n_iteration = len(hist_question)
    agent = model(param=param, n_iteration=n_iteration,
                  **task_param)
    diff = np.zeros(n_iteration)

    log_likelihood = np.zeros(n_iteration)

    for t in range(n_iteration):
        item = hist_question[t]
        p_r = agent.p(item=item)
        # p_choose_correct = min(1, p_r + p_random)
        #
        # s = hist_success[t]
        # p_random = 1 / task_param['n_possible_replies']

        p_random = (1 - p_r) / task_param['n_possible_replies']
        p_choose_correct = p_r + p_random
        s = hist_success[t]

        diff[t] = (s - p_choose_correct)

        lh = p_choose_correct if s else p_random
        log_likelihood[t] = np.log(lh)

        agent.learn(item=item)

    diff = np.power(diff, 2)
    value = np.sum(diff)

    # value = - np.sum(log_likelihood)
    return value
