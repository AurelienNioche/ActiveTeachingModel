import matplotlib.pyplot as plt
import numpy as np


def f(t, t_r, b, a=1):
    print("delta", t-t_r)
    y = a * np.exp(-b*(t-t_r))

    return y


def proof_of_concept():

    success_rate = 0.9

    base_delta = 2

    b = - np.log(success_rate)
    alpha = 1 + np.log(success_rate) / (base_delta*b)

    print(f"b = {b}, alpha={alpha}")

    delta = np.array([(base_delta**n) for n in range(0, 10)])

    for i, d in enumerate(delta):
        print(f"delta={d}; p: {np.exp(-b * d)}")
        b *= (1 - alpha)


def main():

    base_delta = 2
    success_rate = 0.9
    n_iteration = 100

    delta = np.array([(base_delta ** n) for n in range(0, 10)])
    print("delta", delta)

    b = - np.log(success_rate)
    alpha = 1 + np.log(success_rate) / (base_delta*b)
    print(f"b = {b}, alpha={alpha}\n")

    y = []

    current_idx = 0
    d = 0
    pres = []
    for t in range(0, n_iteration):

        p = np.exp(-b*d)
        print(f"t={t}; delta={d}; p={p:.3f}")

        y.append(p)

        if d == delta[current_idx]:
            b *= (1 - alpha)
            d = 0
            current_idx += 1
            pres.append(t)
            print("***")

        else:
            d += 1

    fig, ax = plt.subplots()
    for pr in pres:
        ax.axvline(pr+0.5, linestyle="--", color="0.2",
                   linewidth=0.5)
    ax.plot(y)
    # ax.axhline(0.9, linestyle="--", color="0.2")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, n_iteration)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
