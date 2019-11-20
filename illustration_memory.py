import matplotlib.pyplot as plt
import numpy as np
import os


def get_alpha_beta(success_rate=0.9, base_delta=2):

    beta = - np.log(success_rate)
    alpha = 1 + np.log(success_rate) / (base_delta*beta)

    print(f"beta = {beta}, alpha={alpha}")

    return alpha, beta


def proof_of_concept():

    base_delta = 2
    alpha, beta = get_alpha_beta(base_delta=base_delta)

    delta = np.array([(base_delta**n) for n in range(0, 10)])

    f = beta

    for i, d in enumerate(delta):
        print(f"delta={d}; p: {np.exp(-f * d)}")
        f *= (1 - alpha)


def fig_memory(y, pres, f_name='memory.pdf'):

    fig, ax = plt.subplots()
    for pr in pres:
        ax.axvline(pr+0.5, linestyle="--", color="0.2",
                   linewidth=0.5)
    ax.plot(y)
    # ax.axhline(0.9, linestyle="--", color="0.2")
    ax.set_ylim(0., 1)
    ax.set_xlim(0, len(y))

    ax.set_xlabel("Time")
    ax.set_ylabel("Probability of recall")

    plt.tight_layout()

    fig_folder = os.path.join("fig", "illustration")
    os.makedirs(fig_folder, exist_ok=True)
    plt.savefig(os.path.join(fig_folder, f_name))


def run(alpha, beta, n_iteration, delta):
    f = beta

    y = []

    current_idx = 0
    d = 0
    pres = []
    for t in range(0, n_iteration):

        p = np.exp(-f * d)
        # print(f"t={t}; delta={d}; p={p:.3f}")

        y.append(p)

        if d == delta[current_idx]:
            f *= (1 - alpha)
            d = 0
            current_idx += 1
            pres.append(t)
            # print("***")

        else:
            d += 1

    return pres, y


def main():

    base_delta = 2
    success_rate = 0.9
    n_iteration = 100

    delta = np.array([(base_delta ** n) for n in range(0, 10)])
    print("delta", delta)

    # b = - np.log(success_rate)
    # alpha = 1 + np.log(success_rate) / (base_delta*b)
    # print(f"b = {b}, alpha={alpha}\n")

    alpha, beta = get_alpha_beta(success_rate=success_rate,
                                 base_delta=base_delta)

    pres, y = \
        run(alpha=alpha, beta=beta, delta=delta,
            n_iteration=n_iteration)

    fig_memory(y=y, pres=pres)


def main_irregular():

    np.random.seed(1234)

    n_iteration = 100

    delta = np.array([np.random.randint(20) for _ in range(0, 20)])
    print("delta", delta)

    alpha, beta = get_alpha_beta()
    pres, y = \
        run(alpha=alpha, beta=beta, delta=delta,
            n_iteration=n_iteration)

    fig_memory(y=y, pres=pres, f_name="memory_irregular.pdf")


def main_regular():

    np.random.seed(1234)

    n_iteration = 100

    delta = np.array([int(n_iteration/7) for _ in range(0, 20)])
    print("delta", delta)

    alpha, beta = get_alpha_beta()

    pres, y = \
        run(alpha=alpha, beta=beta, delta=delta,
            n_iteration=n_iteration)

    fig_memory(y=y, pres=pres, f_name="memory_regular.pdf")


if __name__ == "__main__":
    main_regular()
    main()
