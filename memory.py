import matplotlib.pyplot as plt
import numpy as np


def f(t, t_r, b):

    y = np.exp(-b*(t-t_r))

    return y


def main():
    b = 0.02
    t_r = 0
    y=[]
    alpha=0.8
    n_iteration = 10000

    presentation = []
    l = 0.1
    for t in range(10):
        presentation.append(int(5**(10*l)))
        l += 0.10
    print(presentation)
        # np.random.choice(n_iteration, 5, replace=False)

    for t in range(n_iteration):

        if t in presentation:
        # if t%200 == 0:
            t_r = t
            b *= (1-alpha)

        y.append(f(t=t, t_r=t_r, b=b))

    fig, ax = plt.subplots()
    ax.plot(y, linewidth=2)
    for t in presentation:
        ax.axvline(t, linestyle="--", color='0.1', zorder=-10)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, n_iteration)
    plt.show()

if __name__=="__main__":
    main()