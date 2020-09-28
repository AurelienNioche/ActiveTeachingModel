import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


grid_size = 100

x = np.linspace(0, 1, grid_size)
y = np.exp(-10*x)  # stats.beta.pdf(x, a=1.5, b=5)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()

decay = 10
x = np.linspace(0, 1, grid_size)
y = np.exp(-decay*x)
z = y[::-1, None] * y[None, :]

plt.imshow(z, origin='lower')
plt.show()