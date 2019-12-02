import numpy as np
from tqdm import tqdm

from adaptive_teaching.constants import P_RECALL, POST_MEAN, POST_SD, HIST, \
    FORGETTING_RATES


def run(learner_model,
        learner_param,
        engine_model,
        engine_param,
        teacher_model,
        teacher_param,
        task_param, seed):

    n_trial = task_param['n_trial']
    n_item = task_param['n_item']

    param = sorted(learner_model.bounds.keys())

    post_means = {pr: np.zeros(n_trial) for pr in param}
    post_sds = {pr: np.zeros(n_trial) for pr in param}

    p_recall = np.zeros((n_item, n_trial))
    forgetting_rates = np.zeros((n_item, n_trial))

    hist_item = np.zeros(n_trial, dtype=int)
    hist_success = np.zeros(n_trial, dtype=bool)

    # Create learner and engine
    learner = learner_model(param=learner_param, n_item=n_item)
    engine = engine_model(
        teacher_model=teacher_model,
        teacher_param=teacher_param,
        learner_model=learner_model,
        task_param=task_param,
        **engine_param
    )

    np.random.seed(seed)

    for t in tqdm(range(n_trial)):

        # Compute an optimal design for the current trial
        item = engine.get_item()

        # Get a response using the optimal design
        p = learner.p_recall(item=item)

        response = p > np.random.random()

        # Update the engine
        engine.update(item, response)

        # Make the user learn
        learner.update(item=item)

        # Backup the mean/std of post dist
        for i, pr in enumerate(param):
            post_means[pr][t] = engine.post_mean[i]
            post_sds[pr][t] = engine.post_sd[i]

        # Backup prob recall / forgetting rates
        for i in range(n_item):
            p_recall[:, t], forgetting_rates[:, t] = \
                learner.p_recalls_and_forgetting_rates()

        # Backup history
        hist_item[t] = item
        hist_success[t] = response

    return {
        P_RECALL: p_recall,
        POST_MEAN: post_means,
        POST_SD: post_sds,
        HIST: hist_item,
        FORGETTING_RATES: forgetting_rates,
    }
