import numpy as np

from . teacher import Teacher

EPS = np.finfo(np.float).eps

from tqdm import tqdm
from time import time
import datetime


class TeacherHalfLife(Teacher):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.delta = np.zeros(len(self.possible_design), dtype=int)
        self.n_pres_minus_one = np.full(len(self.possible_design), -1, dtype=int)
        self.seen = np.zeros(len(self.possible_design), dtype=bool)

    def _compute_log_lik(self):
        """Compute the log likelihood."""

        t = time()
        tqdm.write("Compute log")

        # log_p = np.zeros((len(self.grid_param), len(self.possible_design)))

        # for j, param in enumerate(self.grid_param):
        #
        #     for i, x in enumerate(self.possible_design):
        #
        #         learner = self.learner_model(
        #             t=len(self.hist),
        #             hist=self.hist,
        #             param=param)
        #         p = learner.p_recall(item=x)
        #         log_p = self.log_lik_bernoulli(self.y, p)
        #         self.log_lik[i, j, :] = log_p
        #
        #         # learner.learn(item=x)
        #         # p_t2 = learner.p_recall(item=x)
        #
        #         # log_p_t2 = self.log_lik_bernoulli(self.y, p_t2)
        #
        #         # self.log_lik_t0_t1[i, j, :] = np.hstack((log_p, log_p_t2))
        #
        # # print(self.log_lik)
        # # old_log_lik = self.log_lik.copy()

        for i in range(len(self.possible_design)):

            p = np.zeros((len(self.grid_param), 2))

            seen = self.seen[i] == 1
            if seen:
                p[:, 1] = np.exp(
                    - self.grid_param[:, 1]
                    * (1-self.grid_param[:, 0])**self.n_pres_minus_one[i]
                    * self.delta[i])
            else:
                p[:, 1] = np.zeros(len(self.grid_param))

            p[:, 0] = 1 - p[:, 1]

            log_p = np.log(p + EPS)

            self.log_lik[i, :, :] = log_p

        # print("# ---------------------- NEW --------------------------- #")
        # print(self.log_lik)
        #
        # assert np.all(self.log_lik == old_log_lik)

            # Learn
            self.n_pres_minus_one[i] += 1
            self.delta += 1

            old_delta = self.delta[i]
            old_seen = self.seen[i]

            self.delta[i] = 1
            self.seen[i] = 1

            new_p = np.zeros((len(self.grid_param), 2))

            seen = self.seen[i] == 1
            if seen:
                new_p[:, 1] = np.exp(
                    - self.grid_param[:, 1]
                    * (1-self.grid_param[:, 0])**self.n_pres_minus_one[i]
                    * self.delta[i])
            else:
                new_p[:, 1] = np.zeros(len(self.grid_param))

            new_p[:, 0] = 1 - new_p[:, 1]

            new_log_p = np.log(new_p + EPS)

            self.log_lik_t0_t1[i, :, :] = np.hstack((log_p, new_log_p))

            self.n_pres_minus_one[i] -= 1
            self.delta -= 1
            self.delta[i] = old_delta
            self.seen[i] = old_seen

        tqdm.write(f"Done! [time elapsed "
        f"{datetime.timedelta(seconds=time() - t)}]")

        # for i, x in enumerate(self.possible_design):
        #
        #     learner = self.learner_model(
        #         t=len(self.hist),
        #         hist=self.hist,
        #         param=param)
        #     p = learner.p_recall(item=x)
        #     learner.learn(item=x)
        #     p_t2 = learner.p_recall(item=x)
        #
        #     log_p = self.log_lik_bernoulli(self.y, p)
        #     log_p_t2 = self.log_lik_bernoulli(self.y, p_t2)
        #
        #     self.log_lik[i, j, :] = log_p
        #     self.log_lik_t0_t1[i, j, :] = np.hstack((log_p, log_p_t2))

    def _update_history(self, design):

        self.n_pres_minus_one[design] += 1
        self.delta += 1
        self.delta[design] = 1
        self.seen[design] = 1

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