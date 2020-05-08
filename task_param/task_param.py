import json
import numpy as np

from learner.learner import Learner

N_SEC_PER_DAY = 86400
N_SEC_PER_ITER = 2


class TaskParam:

    def __init__(self, bounds, param, param_labels, n_item,
                 n_iter_per_ss, n_iter_between_ss,
                 n_ss, thr, mcts_iter_limit, mcts_horizon,
                 seed,):

        self.bounds = np.asarray(bounds)
        self.param = Learner.generate_param(param=param, bounds=bounds,
                                            n_item=n_item)
        self.param_labels = param_labels
        self.n_item = n_item
        self.n_iter_per_ss = n_iter_per_ss

        if isinstance(n_iter_between_ss, str):
            if n_iter_between_ss == 'day':
                n_iter_between_ss = int(
                    N_SEC_PER_DAY/N_SEC_PER_ITER - self.n_iter_per_ss)
            else:
                raise ValueError

        self.n_iter_between_ss = n_iter_between_ss

        self.n_ss = n_ss
        self.thr = thr
        self.mcts_iter_limit = mcts_iter_limit
        self.mcts_horizon = mcts_horizon
        self.seed = seed

        self.n_iter = n_ss * n_iter_per_ss
        self.terminal_t = n_ss * (n_iter_per_ss + n_iter_between_ss)

        n_param = len(self.bounds)
        if self.param.shape == (n_item, n_param):
            pr_str = f"het_param"
        else:
            pr_str = '_'.join(f'{k[0]}={v:.2f}'
                              for (k, v) in zip(param_labels, param))

        self.extension = \
            f'n_ss={n_ss}_' \
            f'n_iter_per_ss={n_iter_per_ss}_' \
            f'n_iter_between_ss={n_iter_between_ss}_' \
            f'mcts_h={mcts_horizon}_' \
            f'{pr_str}_' \
            f'seed={seed}'

        # print("extension:\n", self.extension, "\n")

        if self.param.shape == (n_item, n_param):
            pr_str = f"het. param"  # (bounds={self.bounds})"
        else:
            param_str_list = []
            for (k, v) in zip(self.param_labels, self.param):
                s = '$\\' + k + f"={v:.2f}$"
                param_str_list.append(s)

            pr_str = ', '.join(param_str_list)

        self.info = \
            r'$n_{\mathrm{session}}=' \
            + str(self.n_ss) + '$\n\n' \
            r'$n_{\mathrm{iter\,per\,session}}=' \
            + str(self.n_iter_per_ss) + '$\n' \
            r'$n_{\mathrm{iter\,between\,session}}=' \
            + str(self.n_iter_between_ss) + '$\n\n' \
            r'$\mathrm{MCTS}_{\mathrm{horizon}}=' \
            + str(self.mcts_horizon) + '$\n' \
            r'$\mathrm{MCTS}_{\mathrm{iter\,limit}}=' \
            + str(self.mcts_iter_limit) + '$\n\n' \
            + pr_str + '\n\n' + \
            r'$\mathrm{seed}=' \
            + str(self.seed) + '$'

        # print("info:\n", self.info, "\n")

    @classmethod
    def get(cls, file):
        with open(file) as f:
            tk = cls(**json.load(f))
        return tk
