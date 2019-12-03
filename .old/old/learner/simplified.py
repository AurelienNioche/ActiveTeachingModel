# import numpy as np
#
# from abc import ABC
#
# from . act_r import ActR
#
#
# class ActROneParam(ActR, ABC):
#
#     bounds = 'd', 0.001, 1.0), )
#
#     def __init__(self, param, default_param, *args, **kwargs):
#
#         super().__init__(metaclass=True, *args, **kwargs)
#
#         # Decay parameter
#         self.d = None
#         # Retrieval threshold
#         self.tau = None
#         # Noise in the activation levels
#         self.s = None
#
#         if isinstance(param, list) or isinstance(param, np.ndarray):
#             param = {
#                 t[0]: param[i] for i, t in enumerate(self.bounds)
#             }
#
#         default_param.update(param)
#         self.set_cognitive_parameters(default_param)
#
#         # Short cut
#         self.temp = self.s * np.square(2)
#
#
# class ActRTwoParam(ActR, ABC):
#
#     bounds = (('d', 0.001, 1.0),
#               ('s', 0.001, 5.0)
#               )
#
#     def __init__(self, param, default_param, *args, **kwargs):
#
#         super().__init__(metaclass=True, *args, **kwargs)
#
#         # Decay parameter
#         self.d = None
#         # Retrieval threshold
#         self.tau = None
#         # Noise in the activation levels
#         self.s = None
#
#         if isinstance(param, list) or isinstance(param, np.ndarray):
#             param = {
#                 t[0]: param[i] for i, t in enumerate(self.bounds)
#             }
#
#         default_param.update(param)
#         self.set_cognitive_parameters(default_param)
#
#         # Short cut
#         self.temp = self.s * np.square(2)
