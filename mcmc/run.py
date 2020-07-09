import numpy as np
from tqdm import tqdm
import sys
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

from model import Leitner
from learner import ActR2008
from learner_exp_decay import ExpDecay
from mcmc import MCMC


class Psy:

    def __init__(self):
        pass

    @staticmethod
    def inferred_param(learner, init_guess,
                       hist, success, timestamp, bounds):
        relevant = hist != -1
        r = minimize(learner.inv_log_lik, x0=init_guess,
                     args=(hist[relevant],
                           success[relevant],
                           timestamp[relevant]),
                     bounds=bounds,
                     method='SLSQP')
        return r.x


def main():
    np.random.seed(1234)

    # n_ss = 1
    # ss_n_iter = 1000   # 1000
    # ss_n_iter_between = 0
    n_item = 500
    # time_per_iter = 2
    n_iter = 1000  # n_ss*ss_n_iter

    # -0.704, 0.0786,0.279, 0.177

    # tau/s, 1/s, c, a
    param = [8.956743002544528, 12.72264631043257,  0.279, 0.177]  # 0.279, 0.177]
    init_guess = np.array([1, 1, 1, 1], dtype=float)
    bounds = ((-100.0, 100.0), (0.0, 100.0), (0., 1.), (0., 1.))
    learner = ActR2008(n_item=n_item, n_iter=n_iter,
                       param=param)
    # param = [0.04, 0.1]
    # learner = ExpDecay(param=param, n_iter=n_iter)
    # init_guess = np.array([0.5, 0.5])
    # bounds = ((0, 1), (0, 1))

    teacher = Leitner.create(n_item=n_item, delay_factor=2, delay_min=2)
    psy = Psy()

    hist = np.full(n_iter, -1, dtype=int)
    success = np.zeros(n_iter, dtype=bool)
    timestamp = np.zeros(n_iter, dtype=int)
    inf_param = np.zeros((n_iter, len(param)))

    now = 0
    was_success = None
    previous_ts = None
    item = None
    for itr in tqdm(range(n_iter), file=sys.stdout):

        item = teacher.ask(now=now,
                           last_was_success=was_success,
                           last_time_reply=previous_ts,
                           idx_last_q=item)

        previous_ts = now
        p = learner.p(item=item, now=now)

        was_success = np.random.random() < p
        # print(f"itr={itr}, p={p:.3f}, s={was_success}")
        hist[itr] = item
        success[itr] = was_success
        timestamp[itr] = now

        inf_param[itr] = psy.inferred_param(
            learner=learner,
            init_guess=init_guess,
            hist=hist, success=success,
            timestamp=timestamp,
            bounds=bounds
        )
        learner.update(item=item, now=now)

        time_per_iter = np.random.randint(2, 1000)

        now += time_per_iter

    u, first_p = np.unique(hist, return_index=True)
    a = np.ma.array(success, mask=False)
    a.mask[first_p] = True
    print("mean success", a.mean())
    # data = hist, success, timestamp

    # accepted, rejected = MCMC.run(
    #     likelihood_computer=learner.log_lik,
    #     prior=learner.prior,
    #     param_init=init_guess,
    #     n_iter=1000,
    #     data=data)
    # samples = accepted[int(len(accepted)/2):, :]
    # print(np.mean(samples, axis=0))

    # sns.jointplot(samples[:, 0], samples[:, 1], alpha=0.1)

    # plt.show()
    print("LLS true param", learner.log_lik(
        param=param, success=success, timestamp=timestamp, hist=hist))
    r = minimize(learner.inv_log_lik, x0=init_guess,
                 args=(hist, success, timestamp),
                 bounds=bounds,
                 method='SLSQP')
    print("inferred param", ', '.join([f'{i:.3f}' for i in r.x]))
    print("LLS inferred param", -r.fun)

    fig, axes = plt.subplots(nrows=len(param))

    for i in range(len(param)):
        axes[i].plot(inf_param[:, i])

    plt.show()


if __name__ == "__main__":
    main()
