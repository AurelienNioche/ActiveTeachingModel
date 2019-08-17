import matplotlib.pyplot as plt
import numpy as np


class Sigmoid:
    def __init__(self, x_min, x_max, maximum, midpoint, steepness):

        self.x_min = x_min
        self.x_max = x_max

        self.maximum = maximum
        self.midpoint = midpoint
        self.steepness = steepness

        self.x = np.arange(self.x_min, self.x_max, 0.1)
        self.y = np.zeros_like(self.x)

    def calculate(self):

        L = self.maximum
        k = self.steepness
        x_0 = self.midpoint

        for i in range(self.x.size):
            self.y[i] = (L / (1 + np.exp(-k * (self.x[i] - x_0))))


def plot(sigmoid):
    plt.plot(sigmoid.y)
    plt.title("sigmoid")
    plt.show()


def main():
    sigmoid = Sigmoid(
        x_min=-30,
        x_max=30,
        maximum=1,
        midpoint=0,
        steepness=0.5
    )
    sigmoid.calculate()
    plot(sigmoid)


if __name__ == '__main__':
    main()
