import numpy as np

def f(i):

    return i**2


def f2():
    list_j = list(range(10))
    list_j.remove(3)

def f3():
    a = np.arange(10)
    list_j = a[a != 3]


import timeit
print(timeit.timeit('f3()', number=10000, setup="from __main__ import f3"))
print(timeit.timeit('f2()', number=10000, setup="from __main__ import f2"))