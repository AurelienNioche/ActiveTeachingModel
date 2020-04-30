import os

import numpy as np

from utils.string import param_string

from model.learner.learner import ExponentialForgetting
from new_teacher import Leitner, ThresholdTeacher, MCTSTeacher, \
    GreedyTeacher, FixedNTeacher

from mcts.reward import RewardThreshold, RewardHalfLife, RewardIntegral, \
    RewardAverage, RewardGoal

from plot.mcts import DataFig, make_fig

from tqdm import tqdm
import pickle

SCRIPT_NAME = os.path.basename(__file__).split(".")[0]

FIG_FOLDER = os.path.join("fig", SCRIPT_NAME)
os.makedirs(FIG_FOLDER, exist_ok=True)
PICKLE_FOLDER = os.path.join("pickle", SCRIPT_NAME)
os.makedirs(PICKLE_FOLDER, exist_ok=True)

PARAM = (0.021, 0.44)
PARAM_LABELS = ("alpha", "beta")

N_ITEM = 500


N_ITER_PER_SS = 100
N_ITER_BETWEEN_SS = 1000  # 0  # 43050

N_SS = 2

N_ITER = N_SS * N_ITER_PER_SS

THR = 0.9

MCTS_ITER_LIMIT = 500
MCTS_HORIZON = None

SEED = 0

TERMINAL_T = N_SS * (N_ITER_PER_SS + N_ITER_BETWEEN_SS)

print("TERMINAL T", TERMINAL_T)

# REWARD = RewardGoal(param=PARAM,
#                     n_item=N_ITEM, tau=THR, t_final=(N_ITER_PER_SS+N_ITER_BETWEEN_SS)*N_SS-N_ITER_BETWEEN_SS)


# REWARD = RewardHalfLife(param=PARAM,
#                         n_item=N_ITEM)

# REWARD = RewardAverage(n_item=N_ITEM, param=PARAM)

REWARD = RewardThreshold(n_item=N_ITEM, param=PARAM, tau=THR)

# REWARD = RewardIntegral(param=PARAM, n_item=N_ITEM,
#                         t_final=(N_ITER_PER_SS+N_ITER_BETWEEN_SS)*N_SS-N_ITER_BETWEEN_SS)

CONDITIONS = Leitner.__name__, ThresholdTeacher.__name__, MCTSTeacher.__name__,


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


def make_figure(data):

    condition_labels = data.keys()
    training = np.zeros(TERMINAL_T, dtype=bool)
    training[:] = np.tile([1, ]*N_ITER_PER_SS + [0, ]*N_ITER_BETWEEN_SS, N_SS)
    data_fig = DataFig(condition_labels=condition_labels,
                       training=training, info=info(), threshold=THR,
                       exam=TERMINAL_T)

    for i, cd in enumerate(condition_labels):

        hist = data[cd]

        seen_in_the_end = list(np.unique(hist))

        n_pres = np.zeros(N_ITEM, dtype=int)
        delta = np.zeros(N_ITEM, dtype=int)

        # For the graph
        n_obs = TERMINAL_T
        n_seen = np.zeros(n_obs, dtype=int)
        # objective = np.zeros(n_obs, )
        n_learnt = np.zeros(n_obs, dtype=int)
        p = [[] for _ in seen_in_the_end]

        it = 0
        c_iter_session = 0
        c_between_session = 0
        t_is_teaching = True

        for t in range(TERMINAL_T+1):

            obs_idx = t

            seen = n_pres[:] > 0
            sum_seen = np.sum(seen)

            n_seen[obs_idx] = sum_seen

            if sum_seen > 0:

                seen_t_idx = np.arange(N_ITEM)[seen]

                fr = PARAM[0] * (1 - PARAM[1]) ** (n_pres[seen] - 1)
                p_t = np.exp(-fr * delta[seen])

                for idx_t, item in enumerate(seen_t_idx):
                    idx_in_the_end = seen_in_the_end.index(item)
                    tup = (obs_idx, p_t[idx_t])
                    p[idx_in_the_end].append(tup)

            # objective[obs_idx] = \
            #     REWARD.reward(n_pres=n_pres, delta=delta, t=t, )
            # normalize=False)
            n_learnt[obs_idx] = \
                RewardThreshold(n_item=N_ITEM, tau=THR, param=PARAM)\
                .reward(n_pres=n_pres, delta=delta, normalize=False)

            # Increment delta for all items
            delta[:] += 1
            t += 1

            if t == TERMINAL_T:
                break

            if t_is_teaching:

                action = hist[it]

                n_pres[action] += 1
                # ...except the one for the selected design that equal one
                delta[action] = 1

                c_iter_session += 1

                it += 1

                if c_iter_session >= N_ITER_PER_SS:
                    c_iter_session = 0
                    if N_ITER_BETWEEN_SS > 0:
                        t_is_teaching = False
            else:
                c_between_session += 1
                if c_between_session >= N_ITER_BETWEEN_SS:
                    c_between_session = 0
                    t_is_teaching = True

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

    if Leitner.__name__ in CONDITIONS:
        tqdm.write("Simulating Leitner Teacher")
        data[Leitner.__name__] = Leitner(
            n_item=N_ITEM,
            n_iter_per_ss=N_ITER_PER_SS,
            n_iter_between_ss=N_ITER_BETWEEN_SS, ) \
            .teach(n_iter=N_ITER, learner_model=ExponentialForgetting,
                   learner_kwargs={"param": PARAM,
                                   "n_item": N_ITEM,
                                   "n_iter_per_session": N_ITER_PER_SS,
                                   "n_iter_between_session": N_ITER_BETWEEN_SS},
                   seed=SEED)
    if ThresholdTeacher.__name__ in CONDITIONS:
        tqdm.write("Simulating Threshold Teacher")
        data["threshold"] = ThresholdTeacher(
            n_item=N_ITEM,
            n_iter_per_ss=N_ITER_PER_SS,
            n_iter_between_ss=N_ITER_BETWEEN_SS,
            param=PARAM,
            learnt_threshold=THR,)\
            .teach(n_iter=N_ITER, seed=SEED)

    if MCTSTeacher.__name__ in CONDITIONS:
        tqdm.write("Simulating MCTS Teacher")
        data["mcts"] = MCTSTeacher(
            n_item=N_ITEM,
            n_iter_per_ss=N_ITER_PER_SS,
            n_iter_between_ss=N_ITER_BETWEEN_SS,
            reward=REWARD,
            terminal_t=TERMINAL_T,
            horizon=MCTS_HORIZON,
            iteration_limit=MCTS_ITER_LIMIT,
        ).teach(n_iter=N_ITER)

    # # Simulate Greedy Teacher
    # tqdm.write("Simulating Greedy Teacher")
    # data["greedy"] = GreedyTeacher(
    #     n_item=N_ITEM,
    #     n_iter_per_ss=N_ITER_PER_SS,
    #     n_iter_between_ss=N_ITER_BETWEEN_SS,
    #     reward=REWARD,)\
    #     .teach(n_iter=N_ITER, seed=SEED)

    with open(BKP_FILE, 'wb') as f:
        pickle.dump(data, f)

    return data


def main():
    make_figure(make_data())


if __name__ == "__main__":
    main()
