import numpy as np


class Reward:

    def __init__(self,  n_item, tau):
        self.tau = tau
        self.n_item = n_item

    def reward(self, learner_p_seen, normalize=True):

        n_learnt = np.sum(learner_p_seen > self.tau)
        if normalize:
            return n_learnt / self.n_item
        else:
            return n_learnt
