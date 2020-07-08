import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from teacher import Leitner
from learner import ActR2008
from mcmc import MCMC


class Psy:

    def __init__(self):
        pass

    def update(self, item, now):
        pass

    def inferred_param(self):
        return np.random.random(4)


def main():
    n_ss = 1
    ss_n_iter = 1000
    ss_n_iter_between = 0
    n_item = 100
    time_per_iter = 100

    # tau, s, c, a
    param = [-0.704, 0.0786,  0.279, 0.177]
    init_guess = [0, 1, 1, 1]

    n_iter = n_ss*ss_n_iter

    learner = ActR2008(n_item=n_item, n_iter=n_iter,
                       param=param)
    teacher = Leitner.create(n_item=n_item, delay_factor=2, delay_min=2)
    psy = Psy()

    now = 0

    hist = np.zeros(n_iter, dtype=int)
    success = np.zeros(n_iter, dtype=bool)
    timestamp = np.zeros(n_iter, dtype=int)
    inf_param = np.zeros((n_iter, len(param)))

    itr = 0
    was_success = None
    previous_ts = None
    item = None
    with tqdm(total=n_iter, file=sys.stdout) as pbar:
        for _ in range(n_ss):
            for _ in range(ss_n_iter):

                item = teacher.ask(now=now,
                                   last_was_success=was_success,
                                   last_time_reply=previous_ts,
                                   idx_last_q=item)

                previous_ts = now
                p = learner.p(item=item, now=now)

                was_success = np.random.random() < p
                hist[itr] = item
                success[itr] = was_success
                timestamp[itr] = now

                inf_param[itr] = psy.inferred_param()
                learner.update(item=item, now=now)

                now += time_per_iter
                itr += 1
                pbar.update()

            now += time_per_iter * ss_n_iter_between

    print("mean success", np.mean(success))
    data = hist, success, timestamp

    MCMC.run(likelihood_computer=learner.log_lik,
             prior=learner.prior,
             param_init=init_guess,
             data=data)


if __name__ == "__main__":
    main()
