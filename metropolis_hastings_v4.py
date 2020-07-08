import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

import numpy as np
from . generic import Teacher


class Leitner(Teacher):

    def __init__(self, n_item, delay_factor, delay_min, box, due):

        self.n_item = n_item

        self.delay_factor = delay_factor
        self.delay_min = delay_min

        self.box = box
        self.due = due

    def update_box_and_due_time(self, last_idx,
                                last_was_success, last_time_reply):

        if last_was_success:
            self.box[last_idx] += 1
        else:
            self.box[last_idx] = \
                max(0, self.box[last_idx] - 1)

        delay = self.delay_factor ** self.box[last_idx]
        # Delay is 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 ... minutes
        self.due[last_idx] = \
            last_time_reply + self.delay_min * delay

    def _pickup_item(self, now):

        seen = np.argwhere(np.asarray(self.box) >= 0).flatten()
        n_seen = len(seen)

        if n_seen == self.n_item:
            return np.argmin(self.due)

        else:
            seen__due = np.asarray(self.due)[seen]
            seen__is_due = np.asarray(seen__due) <= now
            if np.sum(seen__is_due):
                seen_and_is_due__due = seen__due[seen__is_due]

                return seen[seen__is_due][np.argmin(seen_and_is_due__due)]
            else:
                return self._pickup_new()

    def _pickup_new(self):
        return np.argmin(self.box)

    def ask(self, now, last_was_success=None, last_time_reply=None,
            idx_last_q=None):

        if idx_last_q is None:
            item_idx = self._pickup_new()

        else:

            self.update_box_and_due_time(
                last_idx=idx_last_q,
                last_was_success=last_was_success,
                last_time_reply=last_time_reply)
            item_idx = self._pickup_item(now)

        return item_idx

    @classmethod
    def create(cls, n_item):
        box = np.full(n_item, -1)
        due = np.full(n_item, -1)
        return cls(n_item=n_item,
                   delay_factor=2,
                   delay_min=2,
                   box=box,
                   due=due)


# The transition model defines how to move from sigma_current to sigma_new
def transition_model(x):
    return x + np.random.normal(size=len(x)) * 0.1


def prior(x):
    # x[0] = mu, x[1]=sigma (new or current)
    # returns 1 for all valid values of sigma. Log(1) =0, so it does not affect the summation.
    # returns 0 for all invalid values of sigma (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
    # It makes the new sigma infinitely unlikely.
    # if (x[1] <= 0):
    #     return 0
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
        return accept < np.exp(x_new - x)


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
        if acceptance_rule(x_lik + np.log(prior(x)),
                           x_new_lik + np.log(prior(x_new))):
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)

    return np.array(accepted), np.array(rejected)


def main():
    n_iter = 5000
    true_x = 10, 2
    observation = np.random.normal(*true_x, size=1000)
    accepted, rejected = \
        metropolis_hastings(
            manual_log_like_normal,
            prior,
            transition_model,
            [0, 1], n_iter,
            observation,
            acceptance)

    colors = ["green", "red"]
    for i, d in enumerate((accepted, )):
        for x, y in d:
            plt.scatter(x, y, color=colors[i], alpha=0.1)
    plt.scatter([true_x[0],], [true_x[1], ], color="blue")
    plt.show()

    burn = len(accepted) // 2
    print(np.mean(accepted[burn:, 0]), np.mean(accepted[burn:, 1]))



if __name__ == "__main__":
    main()
