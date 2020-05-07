import numpy as np
import json

from utils.string import param_string

from teacher import Leitner, ThresholdTeacher, MCTSTeacher, \
    ThresholdPsychologist, MCTSPsychologist

from run.make_fig import make_fig
from run.make_data import make_data


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

        pr_str = \
            param_string(param_labels=param_labels, param=param,
                         first_letter_only=True)

        self.extension = \
                f'n_ss={n_ss}_' \
                f'n_iter_per_ss={n_iter_per_ss}_' \
                f'n_iter_between_ss={n_iter_between_ss}_' \
                f'mcts_h={mcts_horizon}_' \
                f'{pr_str}_' \
                f'seed={seed}'


def main(force=True):
    task_param = TaskParam(**json.load(open('config/task_param.json')))
    teachers = \
        (Leitner, ThresholdTeacher, MCTSTeacher,
         ThresholdPsychologist, MCTSPsychologist)
    make_fig(*make_data(tk=task_param, teachers=teachers,
                        force=force))


if __name__ == "__main__":
    main()
