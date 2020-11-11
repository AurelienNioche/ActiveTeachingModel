import numpy as np


q = np.array([0.5, 0.5])
b = np.array([-1/2, 1/2])

beta_q = 100
beta_b = 0.1
p = np.exp(beta_q * q[0] + beta_b * b[0]) / np.sum(np.exp(beta_q*q + beta_b*b))
print(p)

p2 = 1 / (1 + np.exp(-(beta_q*(q[0] - q[1]) + beta_b*(b[0] - b[1]))))
print(p2)
