import matplotlib.pyplot as plt
import numpy as np
import os


def fig_illustration(f_name='illustration_memorize.pdf'):

    fig, ax = plt.subplots(figsize=(2, 2))
    x = np.linspace(0, 1, 100)
    y = 1-x
    ax.plot(x, y)
    # ax.axhline(0.9, linestyle="--", color="0.2")
    ax.set_ylim(0., 1)
    ax.set_xlim(0, max(x))
    ax.set_yticks((0, 0.5, 1))

    ax.set_xlabel("p")
    ax.set_ylabel("u")

    plt.tight_layout()

    fig_folder = os.path.join("fig", "illustration")
    os.makedirs(fig_folder, exist_ok=True)
    plt.savefig(os.path.join(fig_folder, f_name))


def main():

    fig_illustration()



if __name__ == "__main__":
    main()
