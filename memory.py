import matplotlib.pyplot as plt
import numpy as np


def f(t, t_r, a, b):

    y = a * np.exp(-b*(t-t_r))

    return y


def main():
    a, b = 1, 0.02
    t_r = 0
    y=[]
    alpha=0.5
    n_iteration = 100
    for t in range(n_iteration):

        if t%20 == 0:
            t_r = t
            b *= (1-alpha)

        y.append(f(t=t, t_r=t_r, a=a, b=b))

    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_ylim(0, 1)
    plt.show()

if __name__=="__main__":
    main()