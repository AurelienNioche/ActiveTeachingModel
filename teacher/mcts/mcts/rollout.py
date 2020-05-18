import numpy as np


class Rollout:

    def __init__(self, n_item, tau):
        self.tau = tau
        self.n_item = n_item
        self.items = np.arange(self.n_item)

    def get_action(self, learner_seen):
        # learner_p_seen,
        # n_seen = np.sum(learner_seen)
        #
        # if n_seen == 0:
        #     items_selected = self.items
        #
        # else:
        #
        #     min_p = np.min(learner_p_seen)
        #
        #     if n_seen == self.n_item or min_p <= self.tau:
        #         is_min = learner_p_seen[:] == min_p
        #         items_selected = self.items[learner_seen][is_min]
        #
        #     else:
        #         unseen = np.logical_not(learner_seen)
        #         items_selected = [self.items[unseen][0]]

        # item = np.random.choice(items_selected)
        item = np.random.choice(self.items[learner_seen])
        return item

    def get_possible_actions(self, learner_seen):

        seen = learner_seen
        n_seen = np.sum(seen)
        if n_seen == 0:
            possible_actions = np.arange(1)
        elif n_seen == self.n_item:
            possible_actions = np.arange(self.n_item)
        else:
            already_seen = np.arange(self.n_item)[seen]
            new = np.max(already_seen) + 1
            possible_actions = list(already_seen) + [new]

        return possible_actions
