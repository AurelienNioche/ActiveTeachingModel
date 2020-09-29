import numpy as np

thr = 0.9

hour = 60**2
day = 24*hour

# delta = 2 * day
# print(- np.log(thr) / delta)

# fr = 0.00005
# print(- np.log(thr) / fr / 60)

# idx = np.array([0, 99])
# for v in np.logspace(0.025, 0.00005, 100)[idx]:
#     print(- np.log(thr) / v )


v = 6 * day
print(- np.log(thr) / 1e-06 / day)

# print(- np.log(thr) / 0.0000001/ day)