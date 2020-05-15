import numpy as np


class RolloutThreshold:

    def __init__(self,  n_item=None, tau=0.9):
        self.tau = tau
        self.n_item = n_item

    def action(self, learner, normalize=True):

        p = learner.p_seen()
        n_learnt = np.sum(p > self.tau)
        if normalize:
            return n_learnt / self.n_item
        else:
            return n_learnt