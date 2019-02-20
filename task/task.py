import numpy as np
import json
import itertools as it

try:
    parameters = json.load(open('task/parameters/parameters.json'))
    assert len(parameters)

except Exception:
    parameters = json.load(open('task/parameters/default.json'))

t_max = int(parameters['t_max'])
n = parameters['n']

items = np.arange(n)


# dist = numpy.linalg.norm(a-b)

# Let assume that semantic is 2d and graphic is 2d

# c_graphic = np.zeros((n, n))
# c_semantic = np.zeros((n, n))
#
# for i, j in it.combinations(range(n), r=2):
#
#     g, s = np.random.random(), np.random.random()
#
#     for x, y in [(i, j), (j, i)]:
#
#         c_graphic[x, y] = g
#         c_semantic[x, y] = s
