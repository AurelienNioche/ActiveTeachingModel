import numpy as np

from . metaclass import GenericTeacher


class Threefold(GenericTeacher):

    def __init__(self, task_param, learner_model, confidence_threshold,
                 alpha=0.33, beta=0.33):

        super().__init__(task_param=task_param,
                         confidence_threshold=confidence_threshold,
                         learner_model=learner_model)

        self.learner = learner_model(task_param=task_param)

        assert (alpha + beta) <= 1, \
            "Sum of alpha and beta should be inferior to 1"

        self.alpha = alpha
        self.beta = beta

        self.t = 0

        self.items = np.arange(self.n_item)

    def ask(self, best_param):

        self.learner.set_param(best_param)

        u = np.zeros(self.n_item)

        for i in range(self.n_item):

            # Learn new item
            self.learner.update(item=i, response=None)

            fr_seen, pr_seen = self.learner.forgetting_rate_and_p_seen()

            u[i] = self.alpha * np.mean(pr_seen) \
                - self.beta * 10 * np.mean(fr_seen) \
                + (1 - self.alpha - self.beta) * \
                  (np.sum(self.learner.seen)/self.t)

            # Unlearn item
            self.learner.cancel_update()

        return np.random.choice(self.items[u == np.max(u)])

    def update(self, item, response):

        self.t += 1


    # def _update_learning_value(self):
    #
    #     for i in range(self.n_design):
    #         self.log_lik[i, :, :] = self._log_p(i)
    #
    #         self._update_history(i)
    #
    #         v = np.zeros(self.n_design)
    #
    #         for j in range(self.n_design):
    #
    #             v[j] = logsumexp(self.log_post[:] + self._log_p(j)[:, 1])
    #
    #         self.learning_value[i] = np.sum(v)
    #
    #         self._cancel_last_update_history()
