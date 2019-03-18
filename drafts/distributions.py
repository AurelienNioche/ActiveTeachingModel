import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


x = np.linspace(0, 1, 1000)
s = 4
y = norm.pdf(x, loc=0.5, scale=s) * s
print(y)
print(min(y), max(y))
plt.plot(x, y)
plt.show()