import matplotlib.pyplot as plt
import numpy as np


def u(x, r):
    return x ** (1 - r)


def u2(x, r):
    return (x ** (1 - r)) / (1-r)


def u3(x, r):

    return (1 - np.exp(- r * x)) / r


def main():

    x = np.random.random(size=1000)
    x.sort()
    for r in (0.99, -0.8, -0.3, 0.3, 0.8, 0.99):
        plt.plot(x, u(x, r), label=r)
    plt.plot((0, 1), (0, 1), lw=1, alpha=0.5,
             ls='--', color='black')
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
