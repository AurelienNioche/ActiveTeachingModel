import os

import numpy as np
import matplotlib.pyplot as plt

from adopy import Engine, Model, Task

from tqdm import tqdm


def calculate_prob(x1, x2, b0, b1, b2):
    """A function to compute the probability of a positive response."""

    logit = b0 + x1 * b1 + x2 * b2
    p_obs = 1. / (1 + np.exp(-logit))
    return p_obs


def calculate_prob_3_param(x1, b0, b1):
    """A function to compute the probability of a positive response."""

    logit = b0 + x1 * b1 ** 2
    p_obs = 1. / (1 + np.exp(-logit))

    return p_obs


def run(grid_design, grid_param, true_param, func, num_trial=100):

    np.random.seed(123)

    param_labels = sorted(grid_param.keys())
    design_labels = sorted(grid_design.keys())

    task = Task(name='My New Experiment',  # Name of the task (optional)
                designs=design_labels,  # Labels of design variables
                responses=[0, 1])  # Possible responses

    print('Task object ready!')

    model = Model(name='My Logistic Model',  # Name of the model (optional)
                  params=param_labels,  # Labels of model parameters
                  func=func,
                  task=task
                  )  # A probability function

    print("Model object ready!")

    engine = Engine(model=model,  # a Model object
                    task=task,  # a Task object
                    grid_design=grid_design,  # a grid for design variables
                    grid_param=grid_param)  # a grid for model parameters

    print("Engine object ready!")

    print("Ready to compute!")

    design_types = ['optimal', 'random']

    post_means = {pr: {d: np.zeros(num_trial) for d in design_types} for pr in
                  param_labels}
    post_sds = {pr: {d: np.zeros(num_trial) for d in design_types} for pr in
                param_labels}

    # Run simulations for three designs
    for design_type in design_types:

        print(f"Computing results for design '{design_type}'")

        # Reset the engine as an initial state
        engine.reset()

        for trial in tqdm(range(num_trial)):
            # Compute an optimal design for the current trial
            design = engine.get_design(design_type)

            # Get a response using the optimal design
            p = func(**true_param, **design)
            response = int(p > np.random.random())

            # Update the engine
            engine.update(design, response)

            for i, pr in enumerate(param_labels):
                post_means[pr][design_type][trial] = engine.post_mean[i]
                post_sds[pr][design_type][trial] = engine.post_sd[i]

    fig, axes = plt.subplots(ncols=len(param_labels), figsize=(12, 6))

    colors = [f'C{i}' for i in range(len(param_labels))]

    for i, ax in enumerate(axes):

        for j, dt in enumerate(design_types):
            pr = param_labels[i]

            means = post_means[pr][dt]
            stds = post_sds[pr][dt]

            true_p = true_param[pr]
            ax.axhline(true_p, linestyle='--', color='black',
                       alpha=.2)

            ax.plot(means, color=colors[j], label=dt)
            ax.fill_between(range(num_trial), means - stds,
                            means + stds, alpha=.2, color=colors[j])

            ax.set_title(pr)
            ax.set_xlabel("time")
            ax.set_ylabel(f"value")

    plt.legend()
    plt.tight_layout()

    FIG_FOLDER = os.path.join("fig", "get_started")
    os.makedirs(FIG_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(FIG_FOLDER,
                             f"adaptive_example_{func.__name__}.pdf"))


def main_three_param():

    true_param = {
        'b0': 2.0,
        'b1': 3.0,
    }

    grid_design = {
        'x1': np.linspace(0, 10, 100),    # grid points within [0, 50]
    }

    grid_param = {
        'b0': np.linspace(0, 10, 100),  # grid points within [-5, 5]
        'b1': np.linspace(0, 10, 100),  # grid points within [-5, 5]
    }

    func = calculate_prob_3_param

    num_trial = 200

    run(true_param=true_param, grid_design=grid_design,
        grid_param=grid_param, func=func, num_trial=num_trial)


def main_original():

    print("Preparing...")

    true_param = {
        'b0': 0.5,
        'b1': 1.5,
        'b2': 2.5,
    }

    grid_design = {
        'x1': np.linspace(0, 50, 10),    # 100 grid points within [0, 50]
        'x2': np.linspace(-20, 30, 10),  # 100 grid points within [-20, 30]
    }

    grid_param = {
        'b0': np.linspace(-5, 5, 10),  # 100 grid points within [-5, 5]
        'b1': np.linspace(-5, 5, 10),  # 100 grid points within [-5, 5]
        'b2': np.linspace(-5, 5, 10),  # 100 grid points within [-5, 5]
    }

    func = calculate_prob

    run(true_param=true_param, grid_design=grid_design,
        grid_param=grid_param, func=func)


if __name__ == '__main__':
    main_original()
    main_three_param()
