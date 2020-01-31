import numpy as np
from tqdm import tqdm

import model.simplified.compute
from model.constants \
    import P, FR, P_SEEN, FR_SEEN, POST_MEAN, POST_SD, \
    HIST_ITEM, HIST_SUCCESS, N_SEEN


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

    p = np.zeros((n_item, n_trial))
    fr = np.zeros((n_item, n_trial))
    p_seen = []
    fr_seen = []

    hist_item = np.zeros(n_trial, dtype=int)
    hist_success = np.zeros(n_trial, dtype=bool)

    n_seen = np.zeros(n_trial, dtype=int)

    # Create learner and engine
    learner = learner_model(param=learner_param, task_param=task_param)
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
        p_recall = learner.p(item=item)

        response = p_recall > np.random.random()

        # Update the engine
        engine.update(item=item, response=response)

        # Make the user learn
        learner.update(item=item, response=response)

        # Backup the mean/std of post dist
        for i, pr in enumerate(param):
            post_means[pr][t] = model.simplified.compute.post_mean[i]
            post_sds[pr][t] = model.simplified.compute.post_sd[i]

        # Backup prob recall / forgetting rates
        fr[:, t], p[:, t] = \
            learner.forgetting_rate_and_p_all()

        fr_seen_t, p_seen_t = \
            learner.forgetting_rate_and_p_seen()

        fr_seen.append(fr_seen_t)
        p_seen.append(p_seen_t)

        # Backup history
        n_seen[t] = np.sum(learner.seen)

        hist_item[t] = item
        hist_success[t] = response

    return {
        N_SEEN: n_seen,
        P: p,
        P_SEEN: p_seen,
        FR: fr,
        FR_SEEN: fr_seen,
        POST_MEAN: post_means,
        POST_SD: post_sds,
        HIST_ITEM: hist_item,
        HIST_SUCCESS: hist_success
    }
