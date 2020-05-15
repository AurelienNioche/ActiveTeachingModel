import numpy as np


class RolloutThreshold:

    def __init__(self, n_item, tau):
        self.tau = tau
        self.n_item = n_item
        self.items = np.arange(self.n_item)

    def get_action(self, learner_p_seen, learner_seen):

        n_seen = np.sum(learner_seen)

        if n_seen == 0:
            items_selected = self.items

        else:

            min_p = np.min(learner_p_seen)

            if n_seen == self.n_item or min_p <= self.tau:
                is_min = learner_p_seen[:] == min_p
                items_selected = self.items[learner_seen][is_min]

            else:
                unseen = np.logical_not(learner_seen)
                items_selected = self.items[unseen]

        item = np.random.choice(items_selected)
        return item
