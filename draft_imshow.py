import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
grid_size = 2
n_param = 2
value = np.arange(grid_size**n_param)
alpha = [0.01, 0.01, 0.02, 0.02]
beta = [0.3, 0.5, 0.3, 0.5]

im = ax.imshow(value.reshape((grid_size, grid_size)).T, origin='upper')
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))

ax.set_xticklabels(sorted(np.unique(alpha)))
ax.set_yticklabels(sorted(np.unique(beta)))

ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")
fig.colorbar(im, label='LLS')
plt.show()
