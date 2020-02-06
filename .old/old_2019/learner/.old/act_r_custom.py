
# class ActR2Param:
#
#     def __init__(self, d, d2, tau, s):
#         # Decay parameter
#         self.d = d
#         self.d2 = d2  # Scale parameter
#
#         # Retrieval threshold
#         self.tau = tau
#         # Noise in the activation levels
#         self.s = s


# class ActR2(ActR):
#
#     version = 2.2
#     bounds = ('d', 0.000001, 1.0), ('d2', 0.000001, 1.0), ('tau', -5, 5), \
#              ('s', 0.0000001, 1)
#
#     """
#     Add some scale parameter for time (d2)
#     """
#
#     def __init__(self, tk, param=None, verbose=False, track_p_recall=False):
#
#         if param is None:
#             pass  # ActR is used as abstract class
#         elif type(param) == dict:
#             self.pr = ActR2Param(**param)
#         elif type(param) in (tuple, list, np.ndarray):
#             self.pr = ActR2Param(*param)
#         else:
#             raise Exception(f"Type {type(param)} "
#                             f"is not handled for parameters")
#
#         super().__init__(tk, verbose=verbose, track_p_recall=track_p_recall)
#
#     def _base_level_learning_activation(self, i):
#
#         """The base-level activation measures
#         how much time has elapsed since the jth use:"""
#
#         time_presentation = np.asarray(self.hist == i).nonzero()[0]
#         if not time_presentation.shape[0]:
#             return -np.inf
#         time_elapsed = (self.t - time_presentation) * self.pr.d2
#         return np.log(np.power(time_elapsed, -self.pr.d).sum())

#
# class ActRPlusPlusParam:
#
#     def __init__(self, d, tau, s, g, m, g_mu, g_sigma, m_mu, m_sigma):
#         # Decay parameter
#         self.d = d
#         # Retrieval threshold
#         self.tau = tau
#         # Noise in the activation levels
#         self.s = s
#
#         self.g = g
#         self.m = m
#
#         self.g_mu = g_mu
#         self.g_sigma = g_sigma
#
#         self.m_mu = m_mu
#         self.m_sigma = m_sigma
