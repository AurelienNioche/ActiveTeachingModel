import os

import numpy as np

from new_teacher.leitner import Leitner
from model.learner.learner import ExponentialForgetting
from new_teacher.bruteforce import BruteForceTeacher
from new_teacher.mcts import MCTSTeacher
from new_teacher.threshold import ThresholdTeacher
from new_teacher.adversarial import AdversarialTeacher

from mcts.reward import RewardThreshold

from plot.single import DataFigSingle, fig_single

from tqdm import tqdm

FIG_FOLDER = os.path.join("fig", os.path.basename(__file__).split(".")[0])
os.makedirs(FIG_FOLDER, exist_ok=True)

PARAM = (0.02, 0.2)

N_ITEM = 500


N_ITER_PER_SS = 150
N_ITER_BETWEEN_SS = 43050

N_SS = 60

N_ITER = N_SS * N_ITER_PER_SS

THR = 0.9


def main():

    # Will stock the hist for each method
    hist_all_teachers = {}

    reward = RewardThreshold(n_item=N_ITEM, param=PARAM)

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

    # Simulate mcts
    tqdm.write("Simulating MCTS Teacher")
    h = np.zeros(N_ITER, dtype=int)
    np.random.seed(0)
    teacher = MCTSTeacher(
        iteration_limit=500,
        n_item=N_ITEM,
        reward=reward,
        horizon=10,
        n_iteration_per_session=N_ITER_PER_SS,
        n_iteration_between_session=N_ITER_BETWEEN_SS,
    )
    for t in tqdm(range(N_ITER)):

        action = teacher.ask()
        h[t] = action
        # print("teacher choose", action)
        # print(f"t={t}, action={action}")

    hist_all_teachers["mcts"] = h

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

    # Simulate Leitner
    tqdm.write("Simulating Leitner Teacher")
    h = np.zeros(N_ITER, dtype=int)
    np.random.seed(0)
    teacher = Leitner(n_item=N_ITEM)
    learner = ExponentialForgetting(param=PARAM, n_item=N_ITEM)

    for t in tqdm(range(N_ITER)):
        item = teacher.ask()
        r = learner.recall(item=item)
        teacher.update(item=item, response=r)
        h[t] = item

    hist_all_teachers["leitner"] = h

    # Simulate Threshold Teacher
    tqdm.write("Simulating Threshold Teacher")
    h = np.zeros(N_ITER, dtype=int)
    np.random.seed(0)
    teacher = ThresholdTeacher(n_item=N_ITEM,
                               learnt_threshold=THR)

    for t in tqdm(range(N_ITER)):
        item = teacher.ask(param=PARAM)
        h[t] = item

    hist_all_teachers["threshold"] = h

    data = DataFigSingle(learner_model=ExponentialForgetting,
                         param=PARAM,
                         param_labels=("alpha", "beta"),
                         n_session=1)

    condition_labels = hist_all_teachers.keys()

    for i, cd in enumerate(condition_labels):

        hist = hist_all_teachers[cd]
        # print("hist", hist)

        c_iter_session = 0
        t = 0

        seen_in_the_end = list(np.unique(hist))

        n_pres = np.zeros(N_ITEM, dtype=int)
        delta = np.zeros(N_ITEM, dtype=int)

        # For the graph
        n_seen = np.zeros(N_SS, dtype=int)
        objective = np.zeros(N_SS, )
        p_item = [[] for _ in seen_in_the_end]

        ss_idx = -1

        for it in range(N_ITER):

            # timestamps_until_it = np.asarray(timestamps)
            # hist_until_it = hist[:it]

            seen = n_pres[:] > 0
            sum_seen = np.sum(seen)

            if c_iter_session == (N_ITER_PER_SS-1):
                ss_idx += 1

                n_seen[ss_idx] = sum_seen

                if sum_seen > 0:

                    seen_t_idx = np.arange(N_ITEM)[seen]

                    fr = PARAM[0] * (1 - PARAM[1]) ** (n_pres[seen] - 1)
                    p_t = np.exp(-fr * delta[seen])

                    # item_seen = np.unique(hist_until_it)
                    # item_seen.sort()
                    #
                    # print("item seen", item_seen)
                    # print("p", p_t)

                    for idx_t, item in enumerate(seen_t_idx):
                        idx_in_the_end = seen_in_the_end.index(item)
                        tup = (ss_idx, p_t[idx_t])
                        p_item[idx_in_the_end].append(tup)

                        # if c_iter_session == 0:
                        #     print(idx_t, tup, n_pres[item])

                objective[ss_idx] = reward.reward(n_pres=n_pres, delta=delta, t=t)

            action = hist[it]

            n_pres[action] += 1

            # Increment delta for all items
            delta[:] += 1
            # ...except the one for the selected design that equal one
            delta[action] = 1

            t += 1
            c_iter_session += 1
            if c_iter_session >= N_ITER_PER_SS:
                delta[:] += N_ITER_BETWEEN_SS
                t += N_ITER_BETWEEN_SS
                c_iter_session = 0

        data.add(
            objective=objective,
            n_seen=n_seen,
            p_item=p_item,
            label=cd
        )

    fig_single(data=data, fig_folder=FIG_FOLDER, time_scale=1)


if __name__ == "__main__":
    main()
