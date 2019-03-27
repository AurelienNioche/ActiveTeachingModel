import numpy as np

import matplotlib.pyplot as plt


def main():

    max_ = np.log(1 + 500)

    x = np.linspace(0, 500, num=10000)
    plt.plot(x, np.log(1 + x))
    plt.plot(x, np.log(1 + x) / max_)
    plt.plot(x, np.log((1 + x) / max_))
    plt.show()


if __name__ == "__main__":

    main()
