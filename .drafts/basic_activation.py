import numpy as np

import matplotlib.pyplot as plt


class Learner:

    def __init__(self):
        self.pr = type('', (object,), {'d': 0.5})

        self.a = np.random.choice(np.arange(60), size=300, replace=True)
        self.a = np.ones(300, dtype=int)
        # self.a[299] = 0

        self.time_presentation = [[] for _ in range(60)]
        for i, e in enumerate(self.a):
            self.time_presentation[e].append(i)
        self.t = 300

        # self.a = np.ones(10)
        # self.a[np.array([1, 4, 5])] = 0

    def base_activation(self, i):

        """The base-level activation measures how much time has elapsed since the jth use:"""

        print(self.time_presentation[i])
        # noinspection PyTypeChecker
        sum_a = np.sum([
            (self.t - t_presentation)**(-self.pr.d)
            for t_presentation in self.time_presentation[i]
        ])

        b = np.log(sum_a) if sum_a > 0 else -np.inf
        return b

    def base_activation_2(self, i):

        """The base-level activation measures how much time has elapsed since the jth use:"""
        sum_a = np.sum((self.t - np.asarray(self.a == i).nonzero()[0]) ** (-self.pr.d))
        b = np.log(sum_a) if sum_a > 0 else -np.inf
        return b


def main():

    import time

    l = Learner()

    print(l.base_activation(0))
    print(l.base_activation_2(0))


if __name__ == "__main__":

    main()
