import numpy as np
from tqdm import tqdm

from . plot import create_fig


def run(adaptive_engine, learner_model, true_param,
        possible_design, grid_size, num_trial):

    np.random.seed(123)

    param = sorted(learner_model.bounds.keys())

    learner = learner_model(param=true_param)
    engine = adaptive_engine(
        learner_model=learner_model,
        possible_design=possible_design,
        grid_size=grid_size)

    design_types = ['optimal', 'random']

    post_means = {d: {pr: np.zeros(num_trial)
                      for pr in param}
                  for d in design_types}
    post_sds = {d: {pr: np.zeros(num_trial)
                    for pr in param}
                for d in design_types}

    # Run simulations for three designs
    for design_type in design_types:

        print("Resetting the engine...")

        # Reset the engine as an initial state
        engine.reset()

        print(f"Computing results for design '{design_type}'...")

        for trial in tqdm(range(num_trial)):

            # Compute an optimal design for the current trial
            design = engine.get_design(design_type)

            # Get a response using the optimal design
            p = learner.p_recall(item=design)
            response = int(p > np.random.random())

            # Update the engine
            engine.update(design, response)

            for i, pr in enumerate(param):
                post_means[design_type][pr][trial] = engine.post_mean[i]
                post_sds[design_type][pr][trial] = engine.post_sd[i]

            # Make the user
            learner.learn(item=design)

    create_fig(param=param, design_types=design_types,
               post_means=post_means, post_sds=post_sds,
               true_param=true_param, num_trial=num_trial,
               fig_name=
               f"adaptive_"
               f"{adaptive_engine.__name__}_"
               f"{learner_model.__name__}.pdf")
