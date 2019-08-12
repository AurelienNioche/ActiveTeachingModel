import matplotlib.pyplot as plt

import numpy as np

def f(x):
    return 1/(1+np.exp((x-0.7)/0.1))


x = np.linspace(0, 1, 100)
y = f(x)
plt.plot(x, y)
plt.xlim((-0.01, 1.01))
plt.ylim((-0.01, 1.01))
plt.axvline(0.95, linestyle='dashed', alpha=0.5)
plt.show()