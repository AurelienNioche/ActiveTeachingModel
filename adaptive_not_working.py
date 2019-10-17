import os

import numpy as np
import matplotlib.pyplot as plt

from adopy import Engine, Model, Task

from tqdm import tqdm

from learner.act_r import ActR


class MyModel:

    def __init__(self):
        self.hist = []

    def _p(self, item, d, tau, s):

        hist = self.hist + [item, ]

        learner = ActR(hist=hist, n_iteration=len(hist), param={
            "d": d, "tau": tau, "s": s})
        p_obs = learner.p_recall(item=item)
        return p_obs

    def calculate_prob(self, x, d, tau, s):
        """A function to compute the probability of a positive response."""

        if type(x) == np.ndarray:

            p_obs = np.zeros((x.shape[0], d.shape[1]))

            for i, x in enumerate(x.flatten()):
                for j, (d, tau, s) in enumerate(zip(d.flatten(), tau.flatten(), s.flatten())):
                    p_obs[i, j] = self._p(x, d, tau, s)

        else:
            p_obs = self._p(x, d, tau, s)
        return p_obs


def main():

    print("Preparing...")

    np.random.seed(123)

    params = ['d', 'tau', 's']

    true_param = {
        'd': 0.5354635033901602,
        'tau': -1.0941799361846622,
        's': 0.08348823400597494
    }

    grid_design = {
        'x': np.arange(100),    # 100 grid points within [0, 50]
    }

    grid_size = 20

    ActR.bounds = \
        ('d', 0.001, 1.0), \
        ('tau', -5, 5), \
        ('s', 0.001, 5.0)

    grid_param = {b[0]: np.linspace(b[1], b[2], grid_size)
                  for b in ActR.bounds}

    task = Task(name='My New Experiment',  # Name of the task (optional)
                designs=['x'],    # Labels of design variables
                responses=[0, 1])        # Possible responses

    print('Task is done')

    my_model = MyModel()

    model = Model(name='My Logistic Model',   # Name of the model (optional)
                params=params,  # Labels of model parameters
                func=my_model.calculate_prob,
                  task=task
                  )        # A probability function

    print("Model is done")
    engine = Engine(model=model,              # a Model object
                    task=task,                # a Task object
                    grid_design=grid_design,  # a grid for design variables
                    grid_param=grid_param)    # a grid for model parameters

    print("Engine is done")

    print("Ready to compute")

    design_types = ['optimal', 'random']

    num_trial = 100  # number of trials

    post_means = {pr: {d: np.zeros(num_trial) for d in design_types} for pr in params}
    post_sds = {pr: {d: np.zeros(num_trial) for d in design_types} for pr in params}

    # Run simulations for three designs
    for design_type in design_types:
        # Reset the engine as an initial state
        engine.reset()

        for trial in tqdm(range(num_trial)):

            print(f"Computing results for design '{design_type}'")

            # Compute an optimal design for the current trial
            design = engine.get_design(design_type)

            print(design['x'])
            my_model.hist += design['x']

            # Get a response using the optimal design
            p = my_model.calculate_prob(**true_param, **design)
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
    plt.savefig(os.path.join(FIG_FOLDER, "adaptive_not_working.pdf"))


if __name__ == '__main__':
    main()
