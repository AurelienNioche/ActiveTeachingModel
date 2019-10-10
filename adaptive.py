import numpy as np

from adopy import Engine, Model, Task


def calculate_prob(x1, x2, b0, b1, b2):
    """A function to compute the probability of a positive response."""
    logit = b0 + x1 * b1 + x1 * b2
    p_obs = 1. / (1 + np.exp(-logit))
    return p_obs


def main():

    print("Preparing...")

    true_param = {
        'b0': 50,
        'b1': 25
    }

    grid_design = {
        'x1': np.linspace(0, 50, 100),    # 100 grid points within [0, 50]
        'x2': np.linspace(-20, 30, 100),  # 100 grid points within [-20, 30]
    }

    grid_param = {
        'b0': np.linspace(-5, 5, 100),  # 100 grid points within [-5, 5]
        'b1': np.linspace(-5, 5, 100),  # 100 grid points within [-5, 5]
        'b2': np.linspace(-5, 5, 100),  # 100 grid points within [-5, 5]
    }

    task = Task(name='My New Experiment',  # Name of the task (optional)
                designs = ['x1', 'x2'],    # Labels of design variables
                responses = [0, 1])        # Possible responses

    model = Model(name='My Logistic Model',   # Name of the model (optional)
                params=['b0', 'b1', 'b2'],  # Labels of model parameters
                func=calculate_prob,
                  task=task
                  )        # A probability function

    engine = Engine(model=model,              # a Model object
                    task=task,                # a Task object
                    grid_design=grid_design,  # a grid for design variables
                    grid_param=grid_param)    # a grid for model parameters

    print("Done")

    NUM_TRIAL = 100  # number of trials

    for trial in range(NUM_TRIAL):
        # Compute an optimal design for the current trial
        design = engine.get_design('optimal')
        print(design)

        # Get a response using the optimal design
        p = calculate_prob(**true_param, **design)
        response = int(p > np.random.random())

        # Update the engine
        engine.update(design, response)


if __name__ == '__main__':
    main()
