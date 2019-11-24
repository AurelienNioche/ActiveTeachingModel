import numpy as np
from tqdm import tqdm

from adaptive_design.constants import P_RECALL, POST_MEAN, POST_SD, HIST, \
    FORGETTING_RATES


def run(learner_model,
        learner_param,
        engine_model,
        grid_size,
        design_type, n_trial, n_item, seed):

    print(f"Computing results for design '{design_type}'...")

    param = sorted(learner_model.bounds.keys())

    post_means = {pr: np.zeros(n_trial) for pr in param}
    post_sds = {pr: np.zeros(n_trial) for pr in param}

    p_recall = np.zeros((n_item, n_trial))
    forgetting_rates = np.zeros((n_item, n_trial))

    hist = np.zeros(n_trial, dtype=int)

    # Create learner and engine
    learner = learner_model(param=learner_param, n_item=n_item)
    engine = engine_model(
        design_type=design_type,
        learner_model=learner_model,
        possible_design=np.arange(n_item),  # item => design
        grid_size=grid_size)

    np.random.seed(seed)

    for t in tqdm(range(n_trial)):

        # Compute an optimal design for the current trial
        design = engine.get_design()

        # Get a response using the optimal design
        p = learner.p_recall(item=design)

        response = int(p > np.random.random())

        # Update the engine
        engine.update(design, response)

        # Backup the mean/std of post dist
        for i, pr in enumerate(param):
            post_means[pr][t] = engine.post_mean[i]
            post_sds[pr][t] = engine.post_sd[i]

        # Backup prob recall / forgetting rates
        for i in range(n_item):
            p_recall[:, t], forgetting_rates[:, t] = \
                learner.p_recalls_and_forgetting_rates()

        # Make the user learn
        learner.learn(item=design)

        # Backup history
        hist[t] = design

    return {
        P_RECALL: p_recall,
        POST_MEAN: post_means,
        POST_SD: post_sds,
        HIST: hist,
        FORGETTING_RATES: forgetting_rates,
    }
