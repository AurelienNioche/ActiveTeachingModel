import matplotlib.pyplot as plt
import numpy as np


def f(x, a, s, mean=0):
    return a*np.exp(
        -((x-mean)**2) / (2*s**2)
    )


amplitude = 251615
sigma = 65**0.5

X = np.linspace(start=-100, stop=100, num=1000)

plt.plot(X, f(X, a=amplitude, s=sigma))
plt.show()
