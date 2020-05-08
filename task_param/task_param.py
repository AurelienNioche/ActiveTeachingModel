import json
import numpy as np


class TaskParam:

    def __init__(self, bounds, param, param_labels, n_item,
                 n_iter_per_ss, n_iter_between_ss,
                 n_ss, thr, mcts_iter_limit, mcts_horizon,
                 seed,):

        self.bounds = np.asarray(bounds)
        if isinstance(param, list):
            self.param = np.asarray(param)
        else:
            self.param = param
        self.param_labels = param_labels
        self.n_item = n_item
        self.n_iter_per_ss = n_iter_per_ss
        self.n_iter_between_ss = n_iter_between_ss
        self.n_ss = n_ss
        self.thr = thr
        self.mcts_iter_limit = mcts_iter_limit
        self.mcts_horizon = mcts_horizon
        self.seed = seed

        self.n_iter = n_ss * n_iter_per_ss
        self.terminal_t = n_ss * (n_iter_per_ss + n_iter_between_ss)

        if isinstance(param, str):
            pr_str = param
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

    def info(self):

        if isinstance(self.param, str):
            pr_str = self.param
        else:
            param_str_list = []
            for (k, v) in zip(self.param_labels, self.param):
                s = '$\\' + k + f"={v:.2f}$"
                param_str_list.append(s)

            pr_str = ', '.join(param_str_list)

        return \
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

    @classmethod
    def get(cls, file):
        with open(file) as f:
            tk = cls(**json.load(f))
        return tk
