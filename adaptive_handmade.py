import numpy as np
import matplotlib.pyplot as plt

# from adopy import Engine, Model, Task

from tqdm import tqdm

from learner.act_r import ActR

from scipy.special import logsumexp


class Adaptive:

    def __init__(self):

        # n param
        self.k = 2

        self.grid_size = 5

        self.n_possible_y = 2

        self.n_design = 10

        self.space = \
            tuple([self.n_design, self.n_possible_y, ] + [self.grid_size, ] * self.k)

        self.p_theta = np.random.random(self.space)

    # def p_theta(self):
    #     return np.random.random(self.space)

    def marginal_entropy(self, theta):

        self.p_obs = self._compute_p_obs()
        self.log_lik = ll = self._compute_log_lik()

        lp = np.ones(self.grid_param.shape[0])
        self.log_prior = lp - logsumexp(lp)
        self.log_post = self.log_prior.copy()

        # Calculate the marginal log likelihood.
        lp = expand_multiple_dims(self.log_post, 1, 1)
        mll = logsumexp(self.log_lik + lp, axis=1)
        self.marg_log_lik = mll  # shape (num_design, num_response)

        # Calculate the marginal entropy and conditional entropy.
        self.ent_marg = -np.sum(np.exp(mll) * mll, -1)  # shape (num_designs,)
        self.ent_cond = np.sum(
            self.post * self.ent_obs, axis=1)  # shape (num_designs,)

        # Calculate the mutual information.
        self.mutual_info = self.ent_marg - \
                           self.ent_cond  # shape (num_designs,)

    def conditional_entropy(self):
        pass

    def big_u(self, item):
        pass

    def small_u(self, d_idx, theta_idx, y_idx):

        dist = self.p_theta[d_idx, y_idx]

        if theta_idx == 0:
            dist_theta[:, ]

        for i in range(self.k):
            _sum +=





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
    plt.savefig("fig/fig.pdf")


if __name__ == '__main__':
    main()
