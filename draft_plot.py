import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(np.random.random(100))
ax.set_xlabel("time")
ax.set_ylabel("value")
txt = ax.text(
    -0.1, 0.5, 'colored text in axes coords',
    transform=ax.transAxes,
    horizontalalignment='right',
    verticalalignment='center',
    color='green', fontsize=15)
txt.set_in_layout(True)
plt.tight_layout()
plt.show()
