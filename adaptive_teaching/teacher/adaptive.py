import numpy as np

from . generic import GenericTeacher


def normalize(x):

    max_x = np.max(x)
    min_x = np.min(x)

    d = max_x - min_x
    if d > 0:
        return (x-min_x)/d

    else:
        return np.full(len(x), 0.5)


class Adaptive(GenericTeacher):

    def __init__(self, task_param, learner_model, confidence_threshold,
                 alpha=0.33, beta=0.33):

        super().__init__(task_param=task_param,
                         confidence_threshold=confidence_threshold,
                         learner_model=learner_model)

        self.learner = learner_model(task_param=task_param)

        # assert (alpha + beta) <= 1, \
        #     "Sum of alpha and beta should be inferior to 1"

        self.alpha = alpha
        self.beta = beta

        self.t = 0

        self.items = np.arange(self.n_item)

    def ask(self, best_param):

        self.learner.set_param(best_param)
        if self.t == 0:
            return np.random.choice(self.items)

        fr_seen, pr_seen = self.learner.forgetting_rate_and_p_seen()
        if np.min(pr_seen) < 0.90:
            return np.random.choice(self.items[self.learner.seen][pr_seen < 0.90])
        else:
            return np.random.choice(self.items[np.invert(self.learner.seen)])

    def update(self, item, response):

        self.t += 1
        self.learner.update(item=item, response=response)


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


# self.learner.set_param(best_param)
#
#         vocab = np.zeros(self.n_item)
#         fr = np.zeros(self.n_item)
#
#         for i in range(self.n_item):
#
#             # Learn new item
#             self.learner.update(item=i, response=None)
#
#             fr_seen, pr_seen = self.learner.forgetting_rate_and_p_seen()
#
#             vocab[i] = np.sum(self.learner.seen)
#
#             fr[i] = - np.mean(fr_seen)
#
#             # Unlearn item
#             self.learner.cancel_update()
#
#         vocab = normalize(vocab)
#         fr = normalize(fr)
#
#         u = self.alpha * fr + (1-self.alpha) * vocab
#         sum_u = np.sum(u)
#         if sum_u:
#             u = u/np.sum(u)
#
#         assert np.max(u) <= 1 and np.min(u) >= 0
#
#         # item = np.random.choice(self.items[u == np.max(u)])
#         item = np.random.choice(self.items, p=u)
#         return item