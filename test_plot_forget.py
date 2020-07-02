import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from datetime import datetime


class ExponentialDecay:

    def __init__(self, n_item, param):
        self.n_item = n_item
        self.param = param

        self.n_pres = np.zeros(n_item, dtype=int)
        self.last_pres = np.zeros(n_item, dtype=int)

    def p(self, item, now):
        fr = self.param[0] * (1-self.param[1])**(self.n_pres[item]-1)
        delta = now - self.last_pres[item]
        return np.exp(-fr * delta)

    def update(self, item, timestamp):
        self.n_pres[item] += 1
        self.last_pres[item] = timestamp


class PowerLaw:

    def __init__(self, n_item, param):

        self.param = param

        self.seen = np.zeros(n_item, dtype=bool)
        self.timestamps = np.zeros(n_item, dtype=object)
        self.timestamps[:] = [[] for _ in range(n_item)]

    def p(self, item, now):

        a, d = self.param

        rep = np.asarray(self.timestamps[item])
        n = len(rep)
        if n == 0:
            return 0

        delta = now - rep
        if np.min(delta) == 0:
            return 1

        with np.errstate(over='ignore'):
            m = np.sum(a*np.power(delta, -d))
        p = m / (1+m)
        return p

    def update(self, item, timestamp):

        self.seen[item] = True
        self.timestamps[item].append(timestamp)


class ActR2005:

    def __init__(self, n_item, param):

        self.param = param

        self.seen = np.zeros(n_item, dtype=bool)
        self.timestamps = np.zeros(n_item, dtype=object)
        self.timestamps[:] = [[] for _ in range(n_item)]

    def p(self, item, now):

        tau, s, a = self.param

        rep = np.asarray(self.timestamps[item])
        n = len(rep)
        if n == 0:
            return 0

        delta = now - rep
        if np.min(delta) == 0:
            return 1

        with np.errstate(over='ignore'):
            m = np.log(np.sum(np.power(delta, -a)))
        x = (-tau + m) / s
        p = expit(x)
        return p

    def update(self, item, timestamp):

        self.seen[item] = True
        self.timestamps[item].append(timestamp)


class ActR2008:

    def __init__(self, n_item, param):

        self.param = param

        self.seen = np.zeros(n_item, dtype=bool)
        self.timestamps = np.zeros(n_item, dtype=object)
        self.timestamps[:] = [[] for _ in range(n_item)]

    def p(self, item, now):

        tau, s, c, a = self.param

        rep = np.asarray(self.timestamps[item])
        n = len(rep)
        if n == 0:
            return 0

        delta = now - rep
        if np.min(delta) == 0:
            return 1

        d = np.full(n, a)
        e_m = np.zeros(n)

        for i in range(n):
            if i > 0:
                d[i] += c*e_m[i-1]
            with np.errstate(over='ignore'):
                e_m[i] = np.sum(np.power(delta[:i+1], -d[:i+1]))
        x = (-tau + np.log(e_m[-1])) / s
        p = expit(x)
        return p

    def update(self, item, timestamp):

        self.seen[item] = True
        self.timestamps[item].append(timestamp)


class ActR2008Fast:

    def __init__(self, param):

        self.param = param

        self.hist = []
        self.ts = []

    def p(self, item, now):

        tau, s, c, a = self.param

        rep = np.asarray(self.ts)[np.asarray(self.hist) == item]
        n = len(rep)
        if n == 0:
            return 0

        delta = now - rep
        if np.min(delta) == 0:
            return 1

        d = np.full(n, a)
        e_m = np.zeros(n)

        for i in range(n):
            if i > 0:
                d[i] += c * e_m[i - 1]
            with np.errstate(over='ignore'):
                e_m[i] = np.sum(np.power(delta[:i + 1], -d[:i + 1]))
        x = (-tau + np.log(e_m[-1])) / s
        p = expit(x)
        return p

    def update(self, item, timestamp):
        self.hist.append(item)
        self.ts.append(timestamp)


class ActR2008PA:

    def __init__(self, n_item, param):

        tau, s, c, a = param

        self.cst0 = np.exp(tau/s)
        self.cst1 = - c * np.exp(tau)
        self.min_s = -s
        self.min_inv_s = -1/s
        self.min_a = -a

        self.seen = np.zeros(n_item, dtype=bool)
        self.timestamps = np.zeros(n_item, dtype=object)
        self.timestamps[:] = [[] for _ in range(n_item)]

    def p(self, item, now):

        rep = np.asarray(self.timestamps[item])
        n = len(rep)
        if n == 0:
            return 0

        delta = now - rep
        if np.min(delta) == 0:
            return 1

        tk_sum = delta[0]**self.min_a
        one_minus_p = 1 + self.cst0 * tk_sum**self.min_inv_s
        with np.errstate(invalid='ignore'):
            for i in range(1, n):
                tk_sum += self.cst1 * one_minus_p**self.min_s + self.min_a
                one_minus_p = 1 + self.cst0 * tk_sum**self.min_inv_s

        p = 1/one_minus_p
        return p

    def update(self, item, timestamp):

        self.seen[item] = True
        self.timestamps[item].append(timestamp)


def main():

    tau, s, c, a = [-0.704, 0.0786, 0.279, 0.177]   # [-0.704, 0.0786, 0.279, 0.177]

    t = np.arange(0, 10000, 1)

    rep_mass = list(np.arange(12)) + [5000, ]
    rep_spaced = list(4**np.arange(12)) + [5000, ]
    linestyles = '-', '--'
    labels = "spaced", "mass",

    for k, rep in enumerate((rep_spaced, rep_mass )):

        learners = (
            ActR2008(n_item=1, param=(tau, s, c, a)),
            ActR2008Fast(param=(tau, s, c, a))
        )
        colors = [f'C{i}'for i in range(len(learners))]

        p_recall = [[] for _ in learners]

        for i in t:
            if i in rep:
                for l in learners:
                    dt_ = datetime.now()
                    l.update(item=0, timestamp=i*60)
                    print( l.__class__.__name__, datetime.now()-dt_)
                print()
            for j, l in enumerate(learners):
                p_r = l.p(item=0, now=i*60)
                p_recall[j].append(p_r)

        for j, pr in enumerate(p_recall):
            plt.plot(t, pr,
                     linestyle=linestyles[k],
                     label=learners[j].__class__.__name__ + labels[k],
                     color=colors[j])
    plt.ylabel("p recall")
    plt.xlabel("time (minutes)")
    plt.ylim(0, 1.01)
    plt.legend()
    plt.show()


# def timeclock():
#
#     tau, s, c, a = [-0.704, 0.0786, 0.279, 0.177]
#     from datetime import datetime
#
#     learners = (
#         ActR2008(n_item=1, param=(tau, s, c, a)),
#     )
#
#         a = datetime.now()
#
#         l = ActR2008(n_item=1, param=(tau, s, c, a))
#         for i in range()

if __name__ == "__main__":
    main()
