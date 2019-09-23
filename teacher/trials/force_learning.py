# class ForceLearning(Active):
#     version = 4
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def _update_usefulness(self, agent):
#         """
#         :param agent: agent object (RL, ACT-R, ...) that implements at least
#             the following methods:
#             * p_recall(item): takes index of a question and gives the
#                 probability of recall for the agent in current state.
#             * learn(item): strengthen the association between a kanji and
#                 its meaning.
#             * unlearn(): cancel the effect of the last call of the learn
#                 method.
#         :return None
#
#         Calculate Usefulness of items
#         """
#
#         self.usefulness[:] = 0
#
#         sum_p_recall = np.zeros(self.tk.n_item)
#
#         current_p_recall = np.zeros(self.tk.n_item)
#         for i in range(self.tk.n_item):
#             current_p_recall[i] = agent.p_recall(i)
#
#             agent.learn(i)
#
#             p_recall = np.zeros(self.tk.n_item)
#             for j in range(self.tk.n_item):
#                 p_recall[j] = agent.p_recall(j)
#
#             agent.unlearn()
#             sum_p_recall[i] = np.sum(np.power(p_recall, 2))
#             self.usefulness[i] = np.sum(p_recall > self.learnt_threshold)
#
#         self.usefulness -= current_p_recall > self.learnt_threshold
#         if max(self.usefulness) <= 0:
#             self.usefulness = sum_p_recall
#
#
# class ActivePlus(Active):
#
#     def __init__(self, depth=1, **kwargs):
#
#         super().__init__(**kwargs)
#         self.depth = depth
#
#     def _update_usefulness(self, agent):
#
#         self.usefulness[:] = 0
#
#         current_history = agent.hist.copy()
#         current_t = agent.t
#
#         n_iteration = len(current_history)
#
#         horizon = min((n_iteration, current_t+self.depth))
#         n_repeat = horizon - current_t
#
#         possible_future = \
#             it.product(range(self.tk.n_item), repeat=n_repeat)
#
#         # print("current history", current_history)
#         # print("current t", current_t)
#         #
#         # print("tmax", n_iteration)
#         # print("horizon", horizon)
#         # print("possible futures", list(possible_future))
#
#         for future in possible_future:
#
#             agent.hist[current_t:horizon] = future
#
#             p_recall = np.zeros(self.tk.n_item)
#             for i in range(self.tk.n_item):
#                 p_recall[i] = agent.p_recall(i, time_index=horizon)
#
#             score_future = np.sum(np.power(p_recall, 2))
#             self.usefulness[future[0]] = max(
#                 (self.usefulness[future[0]],
#                  score_future)
#             )
#
#         agent.hist = current_history