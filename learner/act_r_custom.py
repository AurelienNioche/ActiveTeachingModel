import numpy as np
from learner.act_r import ActR
np.seterr(all='raise')


class ActRMeaning(ActR):

    version = 3.1
    bounds = ('d', 0.0000001, 1.0), ('tau', -1, 1), ('s', 0.0000001, 1), \
             ('m', -0.1, 0.1)

    def __init__(self, tk, param=None, **kwargs):

        # Decay parameter
        self.d = None
        # Retrieval threshold
        self.tau = None
        # Noise in the activation levels
        self.s = None

        # Helping from items with close meaning
        self.m = None

        self.set_parameters(param)

        if param is not None:
            self.x = self.m
            self.c_x = tk.c_semantic

        self.items = np.arange(tk.n_item)

        super().__init__(tk=tk, **kwargs)

        print(self.d)

    def p_recall(self, item, time=None, time_index=None):

        # For item i
        pr_effect_i = self._presentation_effect(item,
                                                time=time,
                                                time_index=time_index)
        if not pr_effect_i:
            return 0

        # For connected items
        list_j = self.items[self.items != item]
        pr_effect_j = np.array(
            [self._presentation_effect(j,
                                       time=time,
                                       time_index=time_index
                                       ) for j in list_j])
        contrib = (self.c_x[item, list_j] * pr_effect_j).sum() * self.x

        _sum = pr_effect_i + contrib
        if _sum <= 0:
            return 0

        try:
            base_activation = np.log(_sum)

        except FloatingPointError as e:
            print(pr_effect_j, contrib, pr_effect_i + contrib)
            raise e

        return self._sigmoid_function(base_activation)

# ========================================================================== #


class ActRGraphic(ActRMeaning):

    bounds = ('d', 0.0000001, 1.0), ('tau', -5, 5), ('s', 0.0000001, 1), \
             ('g', -0.1, 0.1)

    def __init__(self, param, tk, verbose=False):

        # Decay parameter
        self.d = None

        # Retrieval threshold
        self.tau = None

        # Noise in the activation levels
        self.s = None

        # Graphic 'help'
        self.g = None

        self.set_parameters(param)

        super().__init__(tk=tk, verbose=verbose)

        self.c_x = self.tk.c_graphic
        self.x = self.g


# ========================================================================== #


class ActRPlusParam:

    def __init__(self, d, tau, s, g, m):

        # Decay parameter
        self.d = d
        # Retrieval threshold
        self.tau = tau
        # Noise in the activation levels
        self.s = s

        self.g = g
        self.m = m


class ActRPlus(ActRMeaning):

    bounds = ('d', 0.0000001, 1.0), ('tau', 0.00, 5), ('s', 0.0000001, 10), \
             ('g', -0.1, 0.1), ('m', -0.1, 0.1)

    def __init__(self, tk, param, verbose=False, track_p_recall=False):

        if type(param) == dict:
            self.pr = ActRPlusParam(**param)
        elif type(param) in (tuple, list, np.ndarray):
            self.pr = ActRPlusParam(*param)
        else:
            raise Exception(f"Type {type(param)} "
                            f"is not handled for parameters")

        super().__init__(tk=tk, verbose=verbose, track_p_recall=track_p_recall)

        self.items = np.arange(self.tk.n_item)

    def p_recall(self, item, time=None, time_index=None):

        # For item i
        pr_effect_i = self._presentation_effect(item,
                                                time=time,
                                                time_index=time_index)
        if not pr_effect_i:
            return 0

        # For connected items
        list_j = self.items[self.items != item]
        pr_effect_j = np.array(
            [self._presentation_effect(j,
                                       time=time,
                                       time_index=time_index
                                       ) for j in list_j])

        semantic_contrib = \
            (self.tk.c_semantic[item, list_j] * pr_effect_j).sum() * self.pr.m
        graphic_contrib = \
            (self.tk.c_graphic[item, list_j] * pr_effect_j).sum() * self.pr.g

        _sum = pr_effect_i + semantic_contrib + graphic_contrib
        if _sum <= 0:
            return 0

        base_activation = np.log(_sum)

        return self._sigmoid_function(base_activation)


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

