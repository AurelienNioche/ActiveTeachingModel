import numpy as np
from tqdm import tqdm

import os

from adaptive_design.engine.teacher_half_life import TeacherHalfLife, \
    RANDOM, OPT_TEACH, OPT_INF0, ADAPTIVE
from adaptive_design.plot import fig_parameter_recovery, \
    fig_p_recall, fig_p_recall_item

from utils.backup import dump, load

from learner.half_life import FastHalfLife

FIG_FOLDER = os.path.join("fig", "adaptive")

P_RECALL = 'p_recall'
POST_MEAN = 'post_mean'
POST_SD = 'post_sd'
HIST = 'hist'
FORGETTING_RATES = 'forgetting_rates'


def run(learner_model,
        learner_param,
        engine_model,
        grid_size,
        design_type, n_trial, n_item):

    print(f"Computing results for design '{design_type}'...")

    param = sorted(learner_model.bounds.keys())

    post_means = {pr: np.zeros(n_trial) for pr in param}
    post_sds = {pr: np.zeros(n_trial) for pr in param}

    p_recall = np.zeros((n_item, n_trial))
    forgetting_rates = np.zeros((n_item, n_trial))

    hist = np.zeros(n_trial)

    # Create learner and engine
    learner = learner_model(param=learner_param, n_item=n_item)
    engine = engine_model(
        design_type=design_type,
        learner_model=learner_model,
        possible_design=np.arange(n_item),  # item => design
        grid_size=grid_size)

    np.random.seed(123)

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


def main():

    force = True, True, True, True

    design_types = [
        OPT_TEACH, OPT_INF0, ADAPTIVE, RANDOM]

    engine_model = TeacherHalfLife

    grid_size = 100
    n_item = 200
    n_trial = 2000

    learner_model = FastHalfLife

    learner_param = {
        "beta": 0.02,
        "alpha": 0.2
    }

    results = {}

    # Run simulations for every design
    for i, dt in enumerate(design_types):

        bkp_file = os.path.join(
            "bkp", "adaptive",
            f"{dt}_"
            f"{learner_model.__name__}_" 
            f"n_trial_{n_trial}_"
            f"n_item_{n_item}"
            f".p")

        r = load(bkp_file)

        if isinstance(force, bool):
            f = force
        else:
            f = force[i]

        if not r or f:
            r = run(
                design_type=dt,
                learner_model=learner_model,
                learner_param=learner_param,
                engine_model=engine_model,
                n_item=n_item,
                n_trial=n_trial,
                grid_size=grid_size
            )

            dump(r, bkp_file)

        results[dt] = r

    post_means = {
        d: results[d][POST_MEAN] for d in design_types
    }

    post_sds = {
        d: results[d][POST_SD] for d in design_types
    }

    p_recall = {
        d: results[d][P_RECALL] for d in design_types
    }

    strength = {
        d: 1/results[d][FORGETTING_RATES] for d in design_types
    }

    param = sorted(learner_model.bounds.keys())

    fig_ext = \
        f"{learner_model.__name__}_" \
        f"n_trial_{n_trial}_" \
        f"n_item_{n_item}" \
        f".pdf"

    fig_name = f"param_recovery_" + fig_ext
    fig_parameter_recovery(param=param, design_types=design_types,
                           post_means=post_means, post_sds=post_sds,
                           true_param=learner_param, num_trial=n_trial,
                           fig_name=fig_name,
                           fig_folder=FIG_FOLDER)

    fig_name = f"p_recall_" + fig_ext
    fig_p_recall(data=p_recall, design_types=design_types,
                 fig_name=fig_name, fig_folder=FIG_FOLDER)

    fig_name = f"p_recall_item_" + fig_ext
    fig_p_recall_item(
        p_recall=p_recall, design_types=design_types,
        fig_name=fig_name, fig_folder=FIG_FOLDER)

    fig_name = f"strength_" + fig_ext
    fig_p_recall(
        y_label="Strength",
        data=strength, design_types=design_types,
        fig_name=fig_name, fig_folder=FIG_FOLDER)


if __name__ == "__main__":
    main()
