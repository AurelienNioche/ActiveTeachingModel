import json
import numpy as np

from model.teacher.leitner import Leitner
from model.teacher.threshold import Threshold
from model.teacher.mcts import MCTSTeacher
from model.teacher.sampling import Sampling

from model.psychologist.psychologist_grid import PsychologistGrid
from model.psychologist.psychologist_gradient import PsychologistGradient

from model.learner.exponential_n_delta import ExponentialNDelta
from model.learner.act_r2005 import ActR2005
from model.learner.act_r2008 import ActR2008
from model.learner.walsh2018 import Walsh2018

N_SEC_PER_DAY = 86400
N_SEC_PER_ITER = 2


TEACHER = {
    'threshold': Threshold,
    'leitner': Leitner,
    'mcts': MCTSTeacher,
    'sampling': Sampling
}

LEARNER = {
    'exp_decay': ExponentialNDelta,
    'act_r2005': ActR2005,
    'act_r2008': ActR2008,
    'walsh2018': Walsh2018
}

PSYCHOLOGIST = {
    'psy_grid': PsychologistGrid,
    'psy_gradient': PsychologistGradient
}


class TaskParam:

    def __init__(self, bounds, param, param_labels, n_item,
                 ss_n_iter, ss_n_iter_between,
                 is_item_specific, time_per_iter,
                 n_ss, thr, mcts,
                 learner_model,
                 psychologist_model,
                 teachers,
                 grid_size,
                 seed, name, leitner,
                 omniscient,
                 init_guess):

        assert len(teachers) == len(omniscient), \
            "'teachers' and 'omniscient' lists should be of equal size"

        self.teachers = []
        for str_t in teachers:
            try:
                t = TEACHER[str_t]
            except KeyError:
                raise ValueError(f'Teacher type not recognized: {str_t}')
            self.teachers.append(t)

        self.learner_model = LEARNER[learner_model]
        self.psychologist_model = PSYCHOLOGIST[psychologist_model]

        self.bounds = np.asarray(bounds)
        if init_guess is not None:
            self.init_guess = np.asarray(init_guess)
        else:
            self.init_guess = np.array([np.mean(b) for b in bounds], dtype=float)

        self.param = self.psychologist_model.generate_param(
            param=param, bounds=bounds,
            n_item=n_item)
        self.param_labels = param_labels
        self.n_item = n_item
        self.ss_n_iter = ss_n_iter

        self.time_per_iter = time_per_iter

        self.is_item_specific = is_item_specific
        self.grid_size = grid_size

        if isinstance(ss_n_iter_between, str):
            if ss_n_iter_between == 'day':
                ss_n_iter_between = int(
                    N_SEC_PER_DAY/N_SEC_PER_ITER - self.ss_n_iter)
            else:
                raise ValueError

        self.ss_n_iter_between = ss_n_iter_between

        self.n_ss = n_ss
        self.learnt_threshold = thr

        self.iter_limit = mcts["iter_limit"]
        self.time_limit = mcts["time_limit"]
        self.horizon = mcts["horizon"]

        self.delay_factor = leitner["delay_factor"]
        self.delay_min = leitner["delay_min"]

        self.omniscient = omniscient

        self.seed = seed

        self.n_iter = n_ss * ss_n_iter
        self.terminal_t = n_ss * (ss_n_iter + ss_n_iter_between)

        n_param = len(self.bounds)
        # if self.param.shape == (n_item, n_param):
        #     pr_str = f"het_param"
        # else:
        #     pr_str = '_'.join(f'{k[0]}={v:.2f}'
        #                       for (k, v) in zip(param_labels, param))

        self.extension = name
            # f'n_ss={n_ss}_' \
            # f'ss_n_iter={ss_n_iter}_' \
            # f'ss_n_iter_between={ss_n_iter_between}_' \
            # f'mcts_h={mcts_horizon}_' \
            # f'{pr_str}_' \
            # f'seed={seed}'

        # print("extension:\n", self.extension, "\n")

        if self.param.shape == (n_item, n_param):
            pr_str = f"het. param"  # (bounds={self.bounds})"
        else:
            param_str_list = []
            for (k, v) in zip(self.param_labels, self.param):
                s = '$' + k + f"={v:.2f}$"
                param_str_list.append(s)

            pr_str = ', '.join(param_str_list)

        mcts_pr_str_list = []
        for (k, v) in mcts.items():
            if isinstance(v, float):
                s = f"{k} = ${v:.2f}$\n"
            else:
                s = f"{k} = {v}\n"
            mcts_pr_str_list.append(s)

        mcts_pr_str = ''.join(mcts_pr_str_list)

        self.info = \
            r'$n_{\mathrm{session}}=' \
            + str(self.n_ss) + '$\n\n' \
            r'$n_{\mathrm{iter\,per\,session}}=' \
            + str(self.ss_n_iter) + '$\n' \
            r'$n_{\mathrm{iter\,between\,session}}=' \
            + str(self.ss_n_iter_between) + '$\n\n' \
            'MCTS:\n' \
            + mcts_pr_str + '\n' \
            + f'Learner: {self.learner_model.__name__}\n' \
            + pr_str + '\n\n' \
            + f'Psychologist: {self.psychologist_model.__name__}\n\n' \
            r'$\mathrm{seed}=' \
            + str(self.seed) + '$'

        # print("info:\n", self.info, "\n")

    @classmethod
    def get(cls, file):
        print(f"I will use the file '{file}'")
        with open(file) as f:
            tk = cls(name=file.split(".")[0].split('/')[-1], **json.load(f))
        return tk
