import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from teacher.psychologist.learner.power_law import PowerLaw

t = np.arange(0, 100, 1)
# learner = PowerLaw(n_item=1)

timestamps = [0]
p_recall = []

d = 0.177     # 0.02
tau = -0.704  # -0.05
temp = 0.0786  # 0.02

rep = [0, 50, ]

learner = PowerLaw(n_item=1)

for i in t:
    if i in rep:
        learner.update(item=0, timestamp=i)

    p_r = learner.p(item=0, param=[d, tau, temp], is_item_specific=False,
                    now=i)
    p_recall.append(p_r)

plt.plot(t, p_recall)
plt.plot(t, np.exp(-0.02*t))
plt.show()
# for i in t:
#     strength = np.log(i ** -d)
#     x = (- tau + strength) / temp
#     # p_r = expit(x)
#     p_r = a / (a+np.exp((-strength + tau)/temp))
#
#     # p_r = learner.p(item=0, param=[1.0, 1, 0, 1], is_item_specific=False,
#     #                 now=i)
#     p_recall.append(p_r)
#
# # t = np.arange(1, 1000)
# #
# # p_recall = 2*(t ** (-0.5))
# # p_recall = p_recall / (1+p_recall)
# plt.plot(p_recall)
#
#
# a = 12
# d = 0.45
#
# p_recall = []
# for i in t:
#     strength = a * i ** -d
#     p_r = strength / (1+strength)
#     p_recall.append(p_r)
#
# plt.plot(p_recall)
#
# p_recall = np.exp(-0.002*t)
# plt.plot(p_recall)
# plt.show()
