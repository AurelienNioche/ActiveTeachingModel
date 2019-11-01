import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from adaptive_design.engine.teacher_half_life import TeacherHalfLife, \
    RANDOM, OPT_TEACH, OPT_INF0, ADAPTIVE
from adaptive_design.plot import create_fig

from utils.utils import dump, load

from learner.half_life import FastHalfLife


P_RECALL = 'p_recall'
POST_MEAN = 'post_mean'
POST_SD = 'post_sd'


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

        for i, pr in enumerate(param):
            post_means[pr][t] = engine.post_mean[i]
            post_sds[pr][t] = engine.post_sd[i]

        for i in range(n_item):
            p_recall[i, t] = learner.p_recall(item=i)

        # Make the user learn
        learner.learn(item=design)

    return {
        P_RECALL: p_recall,
        POST_MEAN: post_means,
        POST_SD: post_sds
    }


def main():

    force = False

    engine_model = TeacherHalfLife

    grid_size = 100
    n_item = 100
    n_trial = 300

    learner_model = FastHalfLife

    learner_param = {
        "beta": 0.02,
        "alpha": 0.2
    }

    design_types = [
        RANDOM, OPT_TEACH, OPT_INF0, ADAPTIVE]

    results = {}

    # Run simulations for every design
    for dt in design_types:

        bkp_file = os.path.join(
            "bkp", "adaptive",
            f"{dt}_"
            f"{learner_model.__name__}_" 
            f"n_trial_{n_trial}_"
            f"n_item_{n_item}"
            f".p")

        r = load(bkp_file)

        if not r or force:
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

    param = sorted(learner_model.bounds.keys())

    fig_ext = \
        f"{learner_model.__name__}_" \
        f"n_trial_{n_trial}_" \
        f"n_item_{n_item}" \
        f".pdf"

    create_fig(param=param, design_types=design_types,
               post_means=post_means, post_sds=post_sds,
               true_param=learner_param, num_trial=n_trial,
               fig_name=f"param_recovery_" + fig_ext)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [f'C{i}' for i in range(len(design_types))]

    for i, dt in enumerate(design_types):

        p_recall = results[dt][P_RECALL]

        means = np.mean(p_recall, axis=0)
        stds = np.std(p_recall, axis=0)

        ax.plot(means, color=colors[i], label=dt)
        ax.fill_between(range(n_trial),
                        means-stds,
                        means+stds,
                        alpha=.2, color=colors[i])

    ax.set_xlabel("time")
    ax.set_ylabel(f"probability or recall")

    plt.legend()
    plt.tight_layout()

    fig_name = f"p_recall_" + fig_ext
    FIG_FOLDER = os.path.join("fig", "adaptive")
    os.makedirs(FIG_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(FIG_FOLDER, fig_name))


if __name__ == "__main__":
    main()
