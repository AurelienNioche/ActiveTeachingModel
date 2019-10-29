import matplotlib.pyplot as plt
import numpy as np


def f(t, t_r, b, a=1):
    print("delta", t-t_r)
    y = a * np.exp(-b*(t-t_r))

    return y


def main():

    success_rate = 0.9

    b = - np.log(success_rate)
    alpha = 1 + np.log(success_rate) / (2*b)
    print(b, alpha)

    pres = np.array([0, 1, ] + [(2**n) for n in range(1, 10)])
    print("pres", pres)

    for i, delta in enumerate(pres):
        if i == 0:
            pass
        else:
            b *= (1 - alpha)
        print(np.exp(-b*delta))

    success_rate = 0.9

    b = - np.log(success_rate)
    alpha = 1 + np.log(success_rate) / (2*b)

    flag = False

    y=[]
    n_iteration = 100

    current_idx = 0
    delta = 0
    for t in range(0, n_iteration):

        if delta == pres[current_idx]:
            print("delta == pres", t)
            print("delta", delta)
            flag = True

        p = np.exp(-b*delta)
        print(f"t={t}; delta={delta}; p={p:.3f}")

        y.append(p)
        delta += 1

        if flag is True:
            if t == 0:
                pass
            else:
                b *= (1 - alpha)
                delta = 0

            current_idx += 1
            flag = False

    fig, ax = plt.subplots()
    ax.plot(y)
    ax.axhline(0.9, linestyle="--", color="0.2")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, n_iteration)
    plt.show()


if __name__ == "__main__":
    main()
