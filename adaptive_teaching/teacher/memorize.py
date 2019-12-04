import numpy as np

from . metaclass import GenericTeacher


class Memorize(GenericTeacher):

    def __init__(self, task_param, learner_model, confidence_threshold=0.1):

        super().__init__(task_param=task_param, learner_model=learner_model,
                         confidence_threshold=confidence_threshold)

        self.items = np.arange(self.n_item)

        self.delta = np.zeros(self.n_item, dtype=int)
        self.n_pres_minus_one = np.full(self.n_item, -1,
                                        dtype=int)
        self.seen = np.zeros(self.n_item, dtype=bool)

        self.learner = learner_model(task_param=task_param)

    def ask(self, best_param):

        if not np.any(self.seen):
            return np.random.choice(self.items)

        self.learner.set_param(best_param)
        fr_seen, pr_seen = self.learner.forgetting_rate_and_p_seen()

        u = 1-pr_seen

        sum_u = np.sum(u)
        if sum_u <= 1 \
                and np.sum(self.seen) < len(self.seen) \
                and np.random.random() < 1-sum_u:
            return np.random.choice(self.items[np.invert(self.seen)])

        else:
            u /= np.sum(u)
            return np.random.choice(self.items[self.seen], p=u)

    def update(self, item, response):

        # Increment delta for all items
        self.delta += 1

        # ...except the one for the selected design that equal one
        self.delta[item] = 1
        self.seen[item] = 1
        self.n_pres_minus_one[item] += 1

        self.learner.update(item=item, response=response)
