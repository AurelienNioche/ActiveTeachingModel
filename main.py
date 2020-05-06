import os

import numpy as np

from utils.string import param_string

from teacher import Leitner, ThresholdTeacher, MCTSTeacher, \
    ThresholdPsychologist, MCTSPsychologist

from learner.learner import Learner
from teacher.mcts.mcts.reward import RewardThreshold
from psychologist.psychologist import Psychologist

from plot.plot import DataFig, make_fig

from tqdm import tqdm
import pickle

SCRIPT_NAME = os.path.basename(__file__).split(".")[0]

FIG_FOLDER = os.path.join("fig", SCRIPT_NAME)
os.makedirs(FIG_FOLDER, exist_ok=True)
PICKLE_FOLDER = os.path.join("pickle", SCRIPT_NAME)
os.makedirs(PICKLE_FOLDER, exist_ok=True)

PARAM = np.array([0.021, 0.44])
PARAM_LABELS = ("alpha", "beta")

N_ITEM = 500


N_ITER_PER_SS = 100  # 2000  # 100 # 150
N_ITER_BETWEEN_SS = 1000  # 1000   # 43050

N_SS = 2

N_ITER = N_SS * N_ITER_PER_SS

THR = 0.9

MCTS_ITER_LIMIT = 500
MCTS_HORIZON = 10   # None

SEED = 0

TERMINAL_T = N_SS * (N_ITER_PER_SS + N_ITER_BETWEEN_SS)

print("TERMINAL T", TERMINAL_T)

REWARD = RewardThreshold(n_item=N_ITEM, tau=THR)

CONDITIONS = (
    Leitner.__name__,
    ThresholdTeacher.__name__,
    ThresholdPsychologist.__name__,
    MCTSPsychologist.__name__,
    MCTSTeacher.__name__
)


PR_STR = \
    param_string(param_labels=PARAM_LABELS, param=PARAM,
                 first_letter_only=True)


EXTENSION = \
        f'n_ss={N_SS}_' \
        f'n_iter_per_ss={N_ITER_PER_SS}_' \
        f'n_iter_between_ss={N_ITER_BETWEEN_SS}_' \
        f'mcts_h={MCTS_HORIZON}_' \
        f'{REWARD.__class__.__name__}_' \
        f'{"_".join(CONDITIONS)}_' \
        f'{PR_STR}_' \
        f'seed={SEED}'

BKP_FILE = os.path.join(PICKLE_FOLDER, f"{EXTENSION}.p")


def info():
    param_str_list = []
    for (k, v) in zip(PARAM_LABELS, PARAM):
        s = '$\\' + k + f"={v:.2f}$"
        param_str_list.append(s)

    param_str = ', '.join(param_str_list)

    return \
        r'$n_{\mathrm{session}}=' + str(N_SS) + '$\n\n' \
        r'$n_{\mathrm{iter\,per\,session}}=' + str(N_ITER_PER_SS) + '$\n' \
        r'$n_{\mathrm{iter\,between\,session}}=' + str(N_ITER_BETWEEN_SS) + '$\n\n' \
        r'$\mathrm{MCTS}_{\mathrm{horizon}}=' + str(MCTS_HORIZON) + '$\n' \
        r'$\mathrm{MCTS}_{\mathrm{iter\,limit}}=' + str(MCTS_ITER_LIMIT) + '$\n\n' \
        + param_str + '\n\n' + \
        r'$\mathrm{seed}=' + str(SEED) + '$'


def make_figure(data, parameter_recovery):

    n_obs = TERMINAL_T
    training = np.zeros(TERMINAL_T, dtype=bool)
    training[:] = np.tile([1, ]*N_ITER_PER_SS + [0, ]*N_ITER_BETWEEN_SS, N_SS)

    cond_labels = list(data.keys())
    cond_labels_param_recovery = list(parameter_recovery.keys()) \
        if parameter_recovery is not None else []

    data_fig = DataFig(cond_labels=cond_labels,
                       training=training, info=info(), threshold=THR,
                       exam=TERMINAL_T,
                       param=PARAM,
                       param_labels=PARAM_LABELS,
                       cond_labels_param_recovery=cond_labels_param_recovery)

    for i, cd in enumerate(cond_labels_param_recovery):
        hist = \
            {
                "post_mean": np.zeros((TERMINAL_T, len(PARAM))),
                "post_std": np.zeros((TERMINAL_T, len(PARAM)))
            }
        c_iter_ss = 0
        for begin_ss in np.arange(0, N_ITER, N_ITER_PER_SS):

            for key in hist.keys():
                split = \
                    parameter_recovery[cd][key][begin_ss:begin_ss+N_ITER_PER_SS]
                last = split[-1, :]
                t = c_iter_ss*(N_ITER_PER_SS+N_ITER_BETWEEN_SS)
                hist[key][t:t+N_ITER_PER_SS] = split
                hist[key][t+N_ITER_PER_SS: t+N_ITER_PER_SS+N_ITER_BETWEEN_SS] = last

            c_iter_ss += 1
        # for t, t_is_teaching in enumerate(training):
        #     if t_is_teaching:
        #         pm = parameter_recovery[cd]['post_mean'][c_iter, :]
        #         psd = parameter_recovery[cd]['post_std'][c_iter, :]
        #         c_iter += 1
        #     hist_pm[t] = pm
        #     hist_psd[t] = psd
        data_fig.add(
            post_mean=hist["post_mean"],
            post_std=hist["post_std"],
        )

    for i, cd in enumerate(cond_labels):

        hist = data[cd]

        seen_in_the_end = list(np.unique(hist))

        n_seen = np.zeros(n_obs, dtype=int)
        n_learnt = np.zeros(n_obs, dtype=int)
        p = [[] for _ in seen_in_the_end]

        it = 0

        learner = Learner(
            n_item=N_ITEM,
            n_iter_per_ss=N_ITER_PER_SS,
            n_iter_between_ss=N_ITER_BETWEEN_SS,
            param=PARAM,
        )

        for t in range(TERMINAL_T):

            t_is_teaching = training[t]

            seen = learner.seen
            sum_seen = np.sum(seen)

            n_seen[t] = sum_seen

            if sum_seen > 0:

                items_seen_at_t = np.arange(N_ITEM)[seen]
                p_at_t = learner.p_seen()

                for item, p_item in zip(items_seen_at_t, p_at_t):
                    idx_in_the_end = seen_in_the_end.index(item)
                    tup = (t, p_item)
                    p[idx_in_the_end].append(tup)

                n_learnt[t] = np.sum(p_at_t > THR)

            item = None
            if t_is_teaching:
                item = hist[it]
                it += 1

            learner.update_one_step_only(item=item)

        data_fig.add(
            n_learnt=n_learnt,
            n_seen=n_seen,
            p=p,
        )

    make_fig(data=data_fig, fig_folder=FIG_FOLDER, fig_name=EXTENSION)


def make_data(force=True):

    if os.path.exists(BKP_FILE) and not force:
        with open(BKP_FILE, 'rb') as f:
            data = pickle.load(f)
        return data

    data = {}
    parameter_recovery = {}

    if Leitner.__name__ in CONDITIONS:
        tqdm.write("Simulating Leitner Teacher")
        learner = Learner(
            n_item=N_ITEM,
            n_iter_per_ss=N_ITER_PER_SS,
            n_iter_between_ss=N_ITER_BETWEEN_SS,
            param=PARAM,
        )
        data[Leitner.__name__] = Leitner(
            n_item=N_ITEM,
            n_iter_per_ss=N_ITER_PER_SS,
            n_iter_between_ss=N_ITER_BETWEEN_SS,
        ).teach(n_iter=N_ITER,
                param=PARAM,
                learner=learner,
                seed=SEED)
    if ThresholdTeacher.__name__ in CONDITIONS:
        tqdm.write("Simulating Threshold Teacher")
        learner = Learner(
            n_item=N_ITEM,
            n_iter_per_ss=N_ITER_PER_SS,
            n_iter_between_ss=N_ITER_BETWEEN_SS,
            param=PARAM,
        )
        teacher = ThresholdTeacher(learner=learner, learnt_threshold=THR)
        data["threshold"] = teacher.teach(n_iter=N_ITER, seed=SEED)

    if ThresholdPsychologist.__name__ in CONDITIONS:
        tqdm.write("Simulating Threshold Teacher + PSY")

        learner = Learner(
            n_item=N_ITEM,
            n_iter_per_ss=N_ITER_PER_SS,
            n_iter_between_ss=N_ITER_BETWEEN_SS,
            param=PARAM,
        )

        psychologist = Psychologist(
            n_iter=N_ITER,
            learner=learner
        )
        teacher = ThresholdPsychologist(
            learner=learner,
            learnt_threshold=THR,
            psychologist=psychologist
        )
        data["threshold_psy"] = teacher.teach(n_iter=N_ITER, seed=SEED)
        parameter_recovery[ThresholdPsychologist.__name__] = {
            'post_mean': teacher.psychologist.hist_pm,
            'post_std': teacher.psychologist.hist_psd}

    if MCTSTeacher.__name__ in CONDITIONS:
        tqdm.write("Simulating MCTS Teacher")
        learner = Learner(
            n_item=N_ITEM,
            n_iter_per_ss=N_ITER_PER_SS,
            n_iter_between_ss=N_ITER_BETWEEN_SS,
            param=PARAM,
        )
        data["mcts"] = MCTSTeacher(
            learner=learner,
            reward=REWARD,
            terminal_t=TERMINAL_T,
            horizon=MCTS_HORIZON,
            iteration_limit=MCTS_ITER_LIMIT,
        ).teach(n_iter=N_ITER)

    if MCTSPsychologist.__name__ in CONDITIONS:
        tqdm.write("Simulating MCTS Teacher + PSY")

        learner = Learner(
            n_item=N_ITEM,
            n_iter_per_ss=N_ITER_PER_SS,
            n_iter_between_ss=N_ITER_BETWEEN_SS,
            param=PARAM,
        )

        psychologist = Psychologist(
            n_iter=N_ITER,
            learner=learner
        )

        teacher = MCTSPsychologist(
            learner=learner,
            psychologist=psychologist,
            reward=REWARD,
            terminal_t=TERMINAL_T,
            horizon=MCTS_HORIZON,
            iteration_limit=MCTS_ITER_LIMIT)
        data["mcts_psy"] = teacher.teach(n_iter=N_ITER, seed=SEED)
        parameter_recovery[MCTSPsychologist.__name__] = {
            'post_mean': teacher.psychologist.hist_pm,
            'post_std': teacher.psychologist.hist_psd}

    with open(BKP_FILE, 'wb') as f:
        pickle.dump((data, parameter_recovery), f)

    return data, parameter_recovery


def main(force=True):
    make_figure(*make_data(force=force))


if __name__ == "__main__":
    main()
