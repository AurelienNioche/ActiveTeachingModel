import numpy as np
from tqdm import tqdm


class MCMC:

    """
    metropolis_hastings
    """
    @staticmethod
    def transition_model(x):
        """
        The transition model defines how to move from sigma_current
        to sigma_new
        """
        return x + np.random.normal(0, 0.01, size=len(x))

    @staticmethod
    def acceptance(x, x_new):
        """
            Defines whether to accept or reject the new sample
        """
        if x_new > x:
            return True
        else:
            accept = np.random.uniform(0, 0.5)
            # Since we did a log likelihood, we need to exponentiate
            # in order to compare to the random number
            # less likely x_new are less likely to be accepted
            q = np.exp(x_new - x)
            return accept < q

    @classmethod
    def run(cls, likelihood_computer, prior, param_init, n_iter, data):
        """
        likelihood_computer(x,data): returns the likelihood that
        # these parameters generated the data
        # transition_model(x): a function that draws a sample from
        # a symmetric distribution and returns it
        # param_init: a starting sample
        # iterations: number of accepted to generated
        # data: the data that we wish to model
        # acceptance_rule(x,x_new): decides whether to accept
        # or reject the new sample
        """
        x = param_init
        accepted = []
        rejected = []
        for _ in tqdm(range(n_iter)):
            x_new = cls.transition_model(x)
            x_lik = likelihood_computer(x, *data)
            x_new_lik = likelihood_computer(x_new, *data)
            if cls.acceptance(
                    x_lik + np.log(prior(x)),
                    x_new_lik + np.log(prior(x_new))):
                x = x_new
                accepted.append(x_new)
            else:
                rejected.append(x_new)

        return np.array(accepted), np.array(rejected)
