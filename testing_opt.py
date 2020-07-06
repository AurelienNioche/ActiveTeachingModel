import numpy as np
from datetime import datetime


# t = datetime.now()

# a = []
# b = np.asarray(a)
# for i in range(10**4):
#     a.append(i)
#     b = np.asarray(i)
#
# print(datetime.now() - t)


seen = np.random.choice([False, True], 500)
t = datetime.now()
for i in range(10**4):
    for j, item in enumerate(np.flatnonzero(seen)):
        pass

print(datetime.now() - t)

seen = np.random.randint(0, 10, 10)
t = datetime.now()
for i in range(10**4):
    for j, item in enumerate(np.flatnonzero(seen)):
        pass

print(datetime.now() - t)