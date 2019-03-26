import numpy as np

P = 0.2
N = 6


def choice():

    r = np.random.random()

    if P > r:
        reply = 0
    else:
        reply = np.random.choice(np.arange(N))

    return reply


def main():

    t_max = 100000

    c = []
    for t in range(t_max):
        c.append(choice())

    for i in range(6):
        label = f'freq(choice={i})'
        print(label, c.count(i)/len(c))

    p_random = 1/N
    p = p_random + 0.2 * (1 - p_random)
    p_other = (1-p) / (N - 1)

    print()
    print("p(choice=0)", p)
    print("p(choice=other than 0)", p_other)


if __name__ == "__main__":

    main()
