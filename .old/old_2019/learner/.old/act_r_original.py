# class ActROriginal(ActR):
# #
# #     def __init__(self, tk, param=None, verbose=False):
# #
# #         super().__init__(tk=tk, param=param, verbose=verbose)
# #
# #     def _base_level_learning_activation(self, i):
# #
# #         """The base-level activation measures how much time has elapsed
# #         since the jth use:"""
# #
# #         # # noinspection PyTypeChecker
# #         # sum_a = np.sum([
# #         #     (self.t - t_presentation)**(-self.pr.d)
# #         #     for t_presentation in self.time_presentation[i]
# #         # ])
# #         #
# #         # b = np.log(sum_a) if sum_a > 0 else -np.inf
# #         # return b
# #         sum_a = np.sum((self.t - np.asarray(self.a == i).nonzero()[0])
#           ** (-self.pr.d))
# #         b = np.log(sum_a) if sum_a > 0 else -np.inf
# #         return b
