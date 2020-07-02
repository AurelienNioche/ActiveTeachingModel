import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# from datetime import datetime

from model_for_testing.act_r2008 import ActR2008


def main():

    # [-0.704, 0.0786, 0.279, 0.177]
    tau, s, c, a = [-0.704, 0.0786, 0.279, 0.177]

    t = np.arange(0, 1000, 1)

    n = 6

    rep_mass = list(np.arange(n))
    linestyles = '-', '--'
    labels = "spaced", "mass",

    counter = 0

    for k, cond in enumerate(labels):

        learners = (
            ActR2008(param=(tau, s, c, a, 60)),
        )
        colors = [f'C{i}'for i in range(len(learners))]

        p_recall = [[] for _ in learners]

        for ts in tqdm(t):
            teach = False
            if cond == "mass":
                if ts in rep_mass:
                    teach = True

            elif cond == "spaced" and ts < 500 and counter < n:
                p = learners[0].p(item=0, now=ts)
                if p < 0.90:
                    counter += 1
                    teach=True

            if teach:
                for le in learners:
                    le.update(item=0, now=ts)
            for j, l in enumerate(learners):
                p_r = l.p(item=0, now=ts)
                p_recall[j].append(p_r)

        for j, pr in enumerate(p_recall):
            plt.plot(t, pr,
                     linestyle=linestyles[k],
                     label=learners[j].__class__.__name__ + labels[k],
                     color=colors[j])

        print(cond, learners[0].ds)
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
