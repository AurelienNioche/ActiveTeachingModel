import os

import numpy as np
import matplotlib.pyplot as plt

from adopy import Engine, Model, Task

from tqdm import tqdm


def calculate_prob(x1, b0, b1):
    """A function to compute the probability of a positive response."""
    #
    # print("x1 shape", x1.shape)
    # print("x2 shape", x2.shape)
    # print("b0 shape", b0.shape)
    # print("b1 shape", b1.shape)
    # print("b2 shape", b2.shape)

    logit = b0 + x1 * b1
    p_obs = 1. / (1 + np.exp(-logit))

    #n print(p_obs.shape)
    return p_obs


def main():

    print("Preparing...")

    np.random.seed(123)

    true_param = {
        'b0': 0.2,
        'b1': 0.5,
    }

    grid_design = {
        'x1': np.linspace(0, 1, 100),    # grid points within [0, 50]
    }

    grid_param = {
        'b0': np.linspace(0, 1, 100),  # grid points within [-5, 5]
        'b1': np.linspace(0, 1, 100),  # grid points within [-5, 5]
    }

    params = sorted(grid_param.keys())

    responses = [0, 1]

    task = Task(name='My New Experiment',  # Name of the task (optional)
                designs=sorted(grid_design.keys()), # Labels of design variables
                responses=responses)        # Possible responses

    print('Task object ready!')

    model = Model(name='My Logistic Model',   # Name of the model (optional)
                params=params,  # Labels of model parameters
                func=calculate_prob,
                  task=task
                  )        # A probability function

    print("Model object ready!")

    engine = Engine(model=model,              # a Model object
                    task=task,                # a Task object
                    grid_design=grid_design,  # a grid for design variables
                    grid_param=grid_param)    # a grid for model parameters

    print("Engine object ready!")

    print("Ready to compute!")

    design_types = ['optimal', 'random']

    num_trial = 1000  # number of trials

    post_means = {pr: {d: np.zeros(num_trial) for d in design_types} for pr in params}
    post_sds = {pr: {d: np.zeros(num_trial) for d in design_types} for pr in params}

    # Run simulations for three designs
    for design_type in design_types:

        print(f"Computing results for design '{design_type}'")

        # Reset the engine as an initial state
        engine.reset()

        for trial in tqdm(range(num_trial)):
            # Compute an optimal design for the current trial
            design = engine.get_design(design_type)

            # Get a response using the optimal design
            p = calculate_prob(**true_param, **design)
            response = int(p > np.random.random())

            # Update the engine
            engine.update(design, response)

            for i, pr in enumerate(params):
                post_means[pr][design_type][trial] = engine.post_mean[i]
                post_sds[pr][design_type][trial] = engine.post_sd[i]

    fig, axes = plt.subplots(ncols=len(params), figsize=(12, 6))

    colors = [f'C{i}' for i in range(len(params))]

    for i, ax in enumerate(axes):

        for j, dt in enumerate(design_types):

            pr = params[i]

            means = post_means[pr][dt]
            stds = post_sds[pr][dt]

            true_p = true_param[pr]
            ax.axhline(true_p, linestyle='--', color='black',
                       alpha=.2)

            ax.plot(means, color=colors[j], label=dt)
            ax.fill_between(range(num_trial), means-stds,
                            means+stds, alpha=.2, color=colors[j])

            ax.set_title(pr)

    plt.legend()

    FIG_FOLDER = os.path.join("fig", "adaptive")
    os.makedirs(FIG_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(FIG_FOLDER, "adaptive_example_2.pdf"))


if __name__ == '__main__':
    main()
