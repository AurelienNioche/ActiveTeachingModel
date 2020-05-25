import numpy as np


class RolloutRandom:

    def __init__(self, n_item, tau):
        self.n_item = n_item
        self.tau = tau
        self.items = np.arange(self.n_item)

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

    def get_action(self, learner_seen, learner_p_seen):

        return np.random.choice(self.get_possible_actions(learner_seen=learner_seen))


class RolloutThreshold(RolloutRandom):

    def __init__(self, n_item, tau):

        super().__init__(n_item=n_item, tau=tau)

    def get_action(self, learner_seen, learner_p_seen):
        n_seen = np.sum(learner_seen)

        if n_seen == 0:
            items_selected = np.arange(1)

        else:

            min_p = np.min(learner_p_seen)

            if n_seen == self.n_item or min_p <= self.tau:
                is_min = learner_p_seen[:] == min_p
                items_selected = self.items[learner_seen][is_min]

            else:
                new = np.max(np.arange(self.n_item)[learner_seen]) + 1
                items_selected = np.array([new, ])

        item = np.random.choice(items_selected)
        # item = np.random.choice(self.items[learner_seen])
        return item


