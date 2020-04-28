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

FIG_FOLDER = os.path.join("fig", os.path.basename(__file__).split(".")[0])
os.makedirs(FIG_FOLDER, exist_ok=True)

PICKLE_FOLDER = os.path.join("pickle", os.path.basename(__file__).split(".")[0])
os.makedirs(PICKLE_FOLDER, exist_ok=True)

PARAM = (0.021, 0.44)
PARAM_LABELS = ("alpha", "beta")

N_ITEM = 500


N_ITER_PER_SS = 100
N_ITER_BETWEEN_SS = 1000  # 43050

N_SS = 2

N_ITER = N_SS * N_ITER_PER_SS

THR = 0.9

MCTS_ITER_LIMIT = 150
# MCTS_HORIZON = 10

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
        f'{REWARD.__class__.__name__}_' \
        f'{"_".join(CONDITIONS)}_' \
        f'{PR_STR}_' \
        f'seed={SEED}'

BKP_FILE = os.path.join(PICKLE_FOLDER, f"{EXTENSION}.p")


def make_figure(data):

    condition_labels = data.keys()
    obs_labels = "t"  # np.array(["start", "end"])
    # obs_criteria = None #np.array([0, N_ITER_PER_SS - 1])

    data_fig = DataFig(condition_labels=condition_labels,
                       observation_labels=obs_labels)

    for i, cd in enumerate(condition_labels):

        hist = data[cd]
        # print("hist", hist)


        # t = 0

        seen_in_the_end = list(np.unique(hist))

        n_pres = np.zeros(N_ITEM, dtype=int)
        delta = np.zeros(N_ITEM, dtype=int)

        # For the graph
        n_obs = TERMINAL_T
        n_seen = {k: np.zeros(n_obs, dtype=int) for k in obs_labels}
        objective = {k: np.zeros(n_obs, ) for k in obs_labels}
        n_learnt = {k: np.zeros(n_obs, dtype=int) for k in obs_labels}
        p = {k: [[] for _ in seen_in_the_end] for k in obs_labels}

        it = 0
        c_iter_session = 0
        c_between_session = 0
        t_is_teaching = True

        # if c_iter_session in obs_criteria:
        k_obs = "t"  # obs_labels[obs_criteria == c_iter_session][0]

        for t in range(TERMINAL_T+1):

            obs_idx = t

            # timestamps_until_it = np.asarray(timestamps)
            # hist_until_it = hist[:it]

            seen = n_pres[:] > 0
            sum_seen = np.sum(seen)

            # ss_idx[k_obs] += 1

            n_seen[k_obs][obs_idx] = sum_seen

            if sum_seen > 0:

                seen_t_idx = np.arange(N_ITEM)[seen]

                fr = PARAM[0] * (1 - PARAM[1]) ** (n_pres[seen] - 1)
                p_t = np.exp(-fr * delta[seen])

                for idx_t, item in enumerate(seen_t_idx):
                    idx_in_the_end = seen_in_the_end.index(item)
                    tup = (obs_idx, p_t[idx_t])
                    p[k_obs][idx_in_the_end].append(tup)

            objective[k_obs][obs_idx] = \
                REWARD.reward(n_pres=n_pres, delta=delta, t=t, )
            # normalize=False)
            n_learnt[k_obs][obs_idx] = \
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
                    # delta[:] += N_ITER_BETWEEN_SS
                    # t += N_ITER_BETWEEN_SS
                    c_iter_session = 0
                    if N_ITER_BETWEEN_SS > 0:
                        t_is_teaching = False
            else:
                c_between_session += 1
                if c_between_session >= N_ITER_BETWEEN_SS:
                    c_between_session = 0
                    t_is_teaching = True

        data_fig.add(
            objective=objective,
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

    # Will stock the hist for each method
    data = {}

    tqdm.write("Simulating Leitner Teacher")
    data["leitner"] = Leitner(
        n_item=N_ITEM,
        n_iter_per_ss=N_ITER_PER_SS,
        n_iter_between_ss=N_ITER_BETWEEN_SS, ) \
        .teach(n_iter=N_ITER, learner_model=ExponentialForgetting,
               learner_kwargs={"param": PARAM,
                               "n_item": N_ITEM,
                               "n_iter_per_session": N_ITER_PER_SS,
                               "n_iter_between_session": N_ITER_BETWEEN_SS},
               seed=SEED)

    tqdm.write("Simulating Threshold Teacher")
    data["threshold"] = ThresholdTeacher(
        n_item=N_ITEM,
        n_iter_per_ss=N_ITER_PER_SS,
        n_iter_between_ss=N_ITER_BETWEEN_SS,
        param=PARAM,
        learnt_threshold=THR,)\
        .teach(n_iter=N_ITER, seed=SEED)

    # Simulate mcts
    tqdm.write("Simulating MCTS Teacher")
    data["mcts"] = MCTSTeacher(
        n_item=N_ITEM,
        n_iter_per_ss=N_ITER_PER_SS,
        n_iter_between_ss=N_ITER_BETWEEN_SS,
        reward=REWARD,
        terminal_t=TERMINAL_T,
        # horizon=MCTS_HORIZON,
        iteration_limit=MCTS_ITER_LIMIT,
    ).teach(n_iter=N_ITER)

    # tqdm.write("Simulating Fixed N Teacher")
    # data["FixedN"] = FixedNTeacher(
    #     n_item=N_ITEM,
    #     n_iter_per_ss=N_ITER_PER_SS,
    #     n_iter_between_ss=N_ITER_BETWEEN_SS)\
    #     .teach(n_iter=N_ITER, seed=SEED)

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

    # # Simulate adversarial
    # tqdm.write("Simulating Adversarial Teacher")
    # h = np.zeros(N_ITER, dtype=int)
    # np.random.seed(0)
    # teacher = AdversarialTeacher(
    #     horizon=0,
    #     param=PARAM,
    #     n_item=N_ITEM,
    #     learnt_threshold=THR)
    # for t in tqdm(range(N_ITER)):
    #
    #     action = teacher.ask()
    #     h[t] = action
    #     # print("teacher choose", action)
    #     # print(f"t={t}, action={action}")
    #
    # hist["adversarial"] = h

    # # Simulate bruteforce
    # tqdm.write("Simulating Bruteforce Teacher")
    # h = np.zeros(N_ITER, dtype=int)
    # np.random.seed(0)
    # teacher = BruteForceTeacher(
    #     horizon=4,
    #     param=PARAM,
    #     n_item=N_ITEM,
    #     learnt_threshold=THR)
    # for t in tqdm(range(N_ITER)):
    #
    #     action = teacher.ask()
    #     h[t] = action
    #     # print("teacher choose", action)
    #     # print(f"t={t}, action={action}")
    #
    # hist["bruteforce"] = h

    # Do the figures

    # learner_model = ExponentialForgetting

    # data_old_method = \
    #     DataFigSingle(learner_model=learner_model,
    #                   param=PARAM, param_labels=PARAM_LABELS,
    #                   n_session=N_SS)
    #
    # c_iter_ss = 0
    # t = 0
    #
    # timestamps = []
    #
    # for it in range(N_ITER):
    #
    #     timestamps.append(t)
    #
    #     t += 1
    #     c_iter_ss += 1
    #     if c_iter_ss >= N_ITER_PER_SS:
    #         t += N_ITER_BETWEEN_SS
    #         c_iter_ss = 0
    #
    # timestamps = np.asarray(timestamps)
    #
    # n_time_steps_per_session = N_ITER_PER_SS + N_ITER_BETWEEN_SS
    # n_time_steps = n_time_steps_per_session * N_SS
    # timesteps = np.arange(0, n_time_steps, n_time_steps_per_session)
    #
    # for i, cd in enumerate(condition_labels):
    #
    #     hist = hist_all_teachers[cd]
    #
    #     d = learner_model().stats_ex_post(
    #         param_labels=PARAM_LABELS,
    #         param=PARAM, hist=hist, timestamps=timestamps,
    #         timesteps=timesteps, learnt_thr=THR,
    #     )
    #
    #     data_old_method.add(
    #         objective=d.n_learnt,
    #         n_seen=d.n_seen,
    #         p_item=d.p_item,
    #         label=cd,
    #         post_mean=d.post_mean,
    #         post_std=d.post_std
    #     )
    #
    # fig_single(data=data_old_method, fig_folder=FIG_FOLDER,
    #            ext='_old_method')


def main():
    make_figure(make_data())


if __name__ == "__main__":
    main()
