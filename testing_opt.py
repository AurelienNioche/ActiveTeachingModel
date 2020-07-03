import numpy as np
from datetime import datetime


t = datetime.now()

a = []
b = np.asarray(a)
for i in range(10**4):
    a.append(i)
    b = np.asarray(i)

print(datetime.now() - t)


t = datetime.now()

a = np.zeros(10**4, dtype=int)
for i in range(10**4):
    a[i] = i

print(datetime.now() - t)