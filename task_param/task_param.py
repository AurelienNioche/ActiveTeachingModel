import json
import numpy as np

from learner.learner import Learner

N_SEC_PER_DAY = 86400
N_SEC_PER_ITER = 2


class TaskParam:

    def __init__(self, bounds, param, param_labels, n_item,
                 n_iter_per_ss, n_iter_between_ss,
                 n_ss, thr, mcts,
                 seed, name):

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
        self.mcts_kwargs = mcts
        self.seed = seed

        self.n_iter = n_ss * n_iter_per_ss
        self.terminal_t = n_ss * (n_iter_per_ss + n_iter_between_ss)

        n_param = len(self.bounds)
        # if self.param.shape == (n_item, n_param):
        #     pr_str = f"het_param"
        # else:
        #     pr_str = '_'.join(f'{k[0]}={v:.2f}'
        #                       for (k, v) in zip(param_labels, param))

        self.extension = name
            # f'n_ss={n_ss}_' \
            # f'n_iter_per_ss={n_iter_per_ss}_' \
            # f'n_iter_between_ss={n_iter_between_ss}_' \
            # f'mcts_h={mcts_horizon}_' \
            # f'{pr_str}_' \
            # f'seed={seed}'

        # print("extension:\n", self.extension, "\n")

        if self.param.shape == (n_item, n_param):
            pr_str = f"het. param"  # (bounds={self.bounds})"
        else:
            param_str_list = []
            for (k, v) in zip(self.param_labels, self.param):
                s = '$\\' + k + f"={v:.2f}$"
                param_str_list.append(s)

            pr_str = ', '.join(param_str_list)

        mcts_pr_str_list = []
        for (k, v) in self.mcts_kwargs.items():
            s = f"{k} = ${v:.2f}$\n"
            mcts_pr_str_list.append(s)

        mcts_pr_str = ', '.join(mcts_pr_str_list)

        self.info = \
            r'$n_{\mathrm{session}}=' \
            + str(self.n_ss) + '$\n\n' \
            r'$n_{\mathrm{iter\,per\,session}}=' \
            + str(self.n_iter_per_ss) + '$\n' \
            r'$n_{\mathrm{iter\,between\,session}}=' \
            + str(self.n_iter_between_ss) + '$\n\n' \
            'MCTS:\n' \
            + mcts_pr_str + '\n' \
            + pr_str + '\n\n' + \
            r'$\mathrm{seed}=' \
            + str(self.seed) + '$'

        # print("info:\n", self.info, "\n")

    @classmethod
    def get(cls, file):
        print(f"I will use the file '{file}'")
        with open(file) as f:
            tk = cls(name=file.split(".")[0].split('/')[-1], **json.load(f))
        return tk
