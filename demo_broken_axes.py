import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import numpy as np

# fig = plt.figure(figsize=(5,2))
bax = brokenaxes(xlims=((0, 0.5), (0.5, 1.0)), ylims=((-1, 0.), (0., 1.)))
x = np.linspace(0, 1, 100)
bax.plot(x, np.sin(10 * x), label='sin')
bax.plot(x, np.cos(10 * x), label='cos')
bax.legend(loc=3)
bax.set_xlabel('time')
bax.set_ylabel('value')
plt.tight_layout()
plt.show()