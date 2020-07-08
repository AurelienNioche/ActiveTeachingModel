import numpy as np
import scipy.stats

# The transition model defines how to move from sigma_current to sigma_new
def transition_model(x):
    return [x[0], np.random.normal(x[1], 0.5, (1,))]


def prior(x):
    # x[0] = mu, x[1]=sigma (new or current)
    # returns 1 for all valid values of sigma. Log(1) =0, so it does not affect the summation.
    # returns 0 for all invalid values of sigma (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
    # It makes the new sigma infinitely unlikely.
    if (x[1] <= 0):
        return 0
    return 1


# Computes the likelihood of the data given a sigma (new or current)
# according to equation (2)
def manual_log_like_normal(x, data):
    # x[0]=mu, x[1]=sigma (new or current)
    # data = the observation
    return np.sum(
        - np.log(x[1] * np.sqrt(2 * np.pi))
        - ((data - x[0]) ** 2) / (2 * x[1] ** 2))


# Same as manual_log_like_normal(x,data), but using scipy implementation.
# It's pretty slow.
def log_lik_normal(x, data):
    # x[0]=mu, x[1]=sigma (new or current)
    # data = the observation
    return np.sum(np.log(scipy.stats.norm(x[0], x[1]).pdf(data)))


# Defines whether to accept or reject the new sample
def acceptance(x, x_new):
    if x_new > x:
        return True
    else:
        accept = np.random.uniform(0, 1)
        # Since we did a log likelihood, we need to exponentiate
        # in order to compare to the random number
        # less likely x_new are less likely to be accepted
        return accept < (np.exp(x_new - x))


def metropolis_hastings(likelihood_computer, prior, transition_model,
                        param_init, iterations, data, acceptance_rule):
    # likelihood_computer(x,data): returns the likelihood that
    # these parameters generated the data
    # transition_model(x): a function that draws a sample from
    # a symmetric distribution and returns it
    # param_init: a starting sample
    # iterations: number of accepted to generated
    # data: the data that we wish to model
    # acceptance_rule(x,x_new): decides whether to accept
    # or reject the new sample
    x = param_init
    accepted = []
    rejected = []
    for i in range(iterations):
        x_new = transition_model(x)
        x_lik = likelihood_computer(x, data)
        x_new_lik = likelihood_computer(x_new, data)
        if (acceptance_rule(x_lik + np.log(prior(x)),
                            x_new_lik + np.log(prior(x_new)))):
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)

    return np.array(accepted), np.array(rejected)


# accepted, rejected = metropolis_hastings(manual_log_like_normal,prior,transition_model,[mu_obs,0.1], 50000,observation,acceptance)
