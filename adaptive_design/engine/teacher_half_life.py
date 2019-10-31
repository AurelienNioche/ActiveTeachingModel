import numpy as np

from . teacher import Teacher


class TeacherHalfLife(Teacher):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.eta = np.zeros(len(self.possible_design))
        self.n_presentation = np.zeros(len(self.possible_design))

    def _compute_log_lik(self):
        """Compute the log likelihood."""

        for j, param in enumerate(self.grid_param):

            for i, x in enumerate(self.possible_design):

                learner = self.learner_model(
                    t=len(self.hist),
                    hist=self.hist,
                    param=param)
                p = learner.p_recall(item=x)
                learner.learn(item=x)
                p_t2 = learner.p_recall(item=x)

                log_p = self.log_lik_bernoulli(self.y, p)
                log_p_t2 = self.log_lik_bernoulli(self.y, p_t2)

                self.log_lik[i, j, :] = log_p
                self.log_lik_t0_t1[i, j, :] = np.hstack((log_p, log_p_t2))

# class HalfLife(Learner):
#
#     version = 2.2
#     bounds = {
#         'alpha': (0.001, 1.0),
#         'beta': (0.001, 1.0),
#     }
#
#     def __init__(
#             self,
#             t=0, hist=None,
#             n_possible_replies=None,
#             param=None,
#             known_param=None,
#             **kwargs):
#
#         super().__init__(**kwargs)
#
#         self.alpha = None
#         self.beta = None
#
#         self.set_cognitive_parameters(param, known_param)
#
#         assert self.alpha is not None
#         assert self.beta is not None
#
#         self.b = {}
#         self.t_r = {}
#
#         if n_possible_replies:
#             self.n_possible_replies = n_possible_replies
#             self.p_random = 1 / self.n_possible_replies
#         else:
#             # raise Exception
#             self.p_random = 0
#
#         if hist is not None:
#             for t_index in range(t):
#
#                 item = hist[t_index]
#                 self.t_r[item] = t_index
#
#                 if item not in self.b:
#                     self.b[item] = self.beta
#                 else:
#                     self.b[item] *= (1 - self.alpha)
#
#         self.t = t
#
#         self.old_b = None
#         self.old_t_r = None
#
#     def p_recall(self, item, time=None, time_index=None):
#
#         if time_index is not None or time is not None:
#             raise NotImplementedError
#
#         if item not in self.t_r:
#             return 0
#
#         b = self.b[item]
#         t_r = self.t_r[item]
#
#         p = np.exp(-b * (self.t - t_r))
#         return p
#
#     def learn(self, item, time=None, time_index=None):
#
#         if time_index is not None or time is not None:
#             raise NotImplementedError
#
#         self.old_b = self.b.copy()
#         self.old_t_r = self.t_r.copy()
#
#         self.t_r[item] = self.t
#
#         if item not in self.b:
#             self.b[item] = self.beta
#         else:
#             self.b[item] *= (1-self.alpha)
#
#         self.t += 1
#
#     def unlearn(self):
#
#         self.b = self.old_b.copy()
#         self.t_r = self.old_t_r.copy()
#
#     def set_history(self, hist, t, times=None):
#         raise NotImplementedError
#
#     def _p_choice(self, item, reply, possible_replies, time=None):
#         raise NotImplementedError
#
#     def _p_correct(self, item, reply, possible_replies, time=None):
#         raise NotImplementedError