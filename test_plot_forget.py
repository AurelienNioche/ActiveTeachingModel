import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from teacher.psychologist.learner.act_r2005 import ActR2005
from teacher.psychologist.learner.act_r import ActR


class PowerLawOriginal:

    def __init__(self, n_item):

        self.seen = np.zeros(n_item, dtype=bool)
        self.timestamps = np.zeros(n_item, dtype=object)
        self.timestamps[:] = [[] for _ in range(n_item)]

        self.m = np.zeros(n_item, dtype=object)
        self.m[:] = [[] for _ in range(n_item)]

    def p(self, item, param, now, is_item_specific):

        tau, s, a = param[item, :] if is_item_specific else param

        rep = np.asarray(self.timestamps[item])
        n = len(rep)
        if n == 0:
            return 0

        delta = now - rep
        if np.min(delta) == 0:
            return 1

        with np.errstate(over='ignore'):
            m = np.log(np.sum(np.power(delta, -a)))
        x = (tau + m) / s
        p = expit(x)
        # print(p)
        return p

    def update(self, item, timestamp):

        self.seen[item] = True
        self.timestamps[item].append(timestamp)


class PowerLawPlus:

    def __init__(self, n_item):

        self.seen = np.zeros(n_item, dtype=bool)
        self.timestamps = np.zeros(n_item, dtype=object)
        self.timestamps[:] = [[] for _ in range(n_item)]

    def p(self, item, param, now, is_item_specific):

        tau, s, c, a = param[item, :] if is_item_specific else param

        rep = np.asarray(self.timestamps[item])
        n = len(rep)
        if n == 0:
            return 0

        delta = now - rep
        if np.min(delta) == 0:
            return 1

        d = np.full(n, a)
        m = np.zeros(n)

        for i in range(n):
            if i > 0:
                d[i] += c*np.exp(m[i-1])
            print("d", d[i])
            with np.errstate(over='ignore'):
                m[i] = np.log(np.sum(np.power(delta[:i+1]/60, -d[:i+1])))
            print("m", m[i])
        x = (tau + m[-1]) / s
        p = expit(x)
        # print(p)
        return p

    def update(self, item, timestamp):

        self.seen[item] = True
        self.timestamps[item].append(timestamp)


def main():
    t = np.arange(0, 1000, 1)
    # learner = ActR2005(n_item=1)

    #
    # param = -0.704, 0.0786, 0.279,  0.177,
    # tau, s, c, a = param

    param = [
        [-0.704, 0.0786, 0.279, 0.177],
        [-0.704, 0.0786, 0.279, 0.177],
        # [-0.704, 0.0786, 0.177],

    ]
    rep =np.arange(0, 500, 20)

    learners = (ActR(n_item=1), PowerLawPlus(n_item=1))
    p_recall = [[] for _ in learners]

    for i in t:
        if i in rep:
            for l in learners:
                l.update(item=0, timestamp=i)
        for j, l in enumerate(learners):
            p_r = l.p(item=0, param=param[j], is_item_specific=False,
                      now=i)
            p_recall[j].append(p_r)

    for j, pr in enumerate(p_recall):
        plt.plot(t, pr, label=learners[j].__class__.__name__)
    plt.plot(t, np.exp(-0.02*t))
    plt.legend()
    plt.show()
    # for i in t:
    #     strength = np.log(i ** -d)
    #     x = (- tau + strength) / temp
    #     # p_r = expit(x)
    #     p_r = a / (a+np.exp((-strength + tau)/temp))
    #
    #     # p_r = learner.p(item=0, param=[1.0, 1, 0, 1], is_item_specific=False,
    #     #                 now=i)
    #     p_recall.append(p_r)
    #
    # # t = np.arange(1, 1000)
    # #
    # # p_recall = 2*(t ** (-0.5))
    # # p_recall = p_recall / (1+p_recall)
    # plt.plot(p_recall)
    #
    #
    # a = 12
    # d = 0.45
    #
    # p_recall = []
    # for i in t:
    #     strength = a * i ** -d
    #     p_r = strength / (1+strength)
    #     p_recall.append(p_r)
    #
    # plt.plot(p_recall)
    #
    # p_recall = np.exp(-0.002*t)
    # plt.plot(p_recall)
    # plt.show()


if __name__ == "__main__":
    main()
