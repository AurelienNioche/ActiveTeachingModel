import json


class TaskParam:

    def __init__(self, param, param_labels, n_item,
                 n_iter_per_ss, n_iter_between_ss,
                 n_ss, thr, mcts_iter_limit, mcts_horizon,
                 seed, ):

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

        pr_str = '_'.join(f'{k[0]}={v:.2f}'
                          for (k, v) in zip(param_labels, param))

        self.extension = \
            f'n_ss={n_ss}_' \
            f'n_iter_per_ss={n_iter_per_ss}_' \
            f'n_iter_between_ss={n_iter_between_ss}_' \
            f'mcts_h={mcts_horizon}_' \
            f'{pr_str}_' \
            f'seed={seed}'

    @classmethod
    def get(cls, file):
        with open(file) as f:
            tk = cls(**json.load(f))
        return tk
