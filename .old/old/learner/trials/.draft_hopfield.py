import numpy as np

N = 10000
P = 16
f = 0.1

kappa = 13000

tau_0 = 1

representation_memory = \
    np.random.choice([0, 1], p=[f, 1-f], size=(P, N))

eta_neuron0 = representation_memory[:, 0]

eta_memory1_neuron0 = representation_memory[1, 0]

print(f'Neuron 0 is encoding memory 1: {eta_memory1_neuron0==1}')


def sinusoid(x, freq):
    return np.sin(2 * np.pi * freq * x + (np.pi / 2)) * np.cos(
        np.pi * x + (np.pi / 2))


def phi(t):
    return sinusoid(t, freq=tau_0)


def J(i, j, t):

    sum_ = 0

    for mu in range(P):
        sum_ += (representation_memory[mu, i] - f) \
            * (representation_memory[mu, j] - f) \
            - phi(t)

    return (kappa/N) * sum_
