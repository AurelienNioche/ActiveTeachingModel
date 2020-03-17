import os

import numpy as np

from model.teacher.leitner import Leitner
from model.learner.learner import ExponentialForgetting
from new_teacher.bruteforce import BruteForceTeacher
from new_teacher.mcts import MCTSTeacher
from new_teacher.threshold import ThresholdTeacher
from new_teacher.adversarial import AdversarialTeacher

from plot.single import DataFigSingle, fig_single

from tqdm import tqdm

FIG_FOLDER = os.path.join("fig", os.path.basename(__file__).split(".")[0])
os.makedirs(FIG_FOLDER, exist_ok=True)

N_ITEM = 100
PARAM = (0.02, 0.2)
THR = 0.9
N_ITER = 100


def main():

    # Will stock the hist for each method
    hist = {}

    # Simulate adversarial
    h = np.zeros(N_ITER, dtype=int)
    np.random.seed(0)
    teacher = AdversarialTeacher(param=PARAM,
                          n_item=N_ITEM,
                          learnt_threshold=THR)
    for t in tqdm(range(N_ITER)):

        action = teacher.ask()
        h[t] = action
        # print("teacher choose", action)
        # print(f"t={t}, action={action}")

    hist["adversarial"] = h

    # Simulate mcts
    h = np.zeros(N_ITER, dtype=int)
    np.random.seed(0)
    teacher = MCTSTeacher(param=PARAM,
                          n_item=N_ITEM,
                          learnt_threshold=THR)
    for t in tqdm(range(N_ITER)):

        action = teacher.ask()
        h[t] = action
        # print("teacher choose", action)
        # print(f"t={t}, action={action}")

    hist["mcts"] = h

    # Simulate bruteforce
    h = np.zeros(N_ITER, dtype=int)
    np.random.seed(0)
    teacher = BruteForceTeacher(param=PARAM,
                                n_item=N_ITEM,
                                learnt_threshold=THR)
    for t in tqdm(range(N_ITER)):

        action = teacher.ask()
        h[t] = action
        # print("teacher choose", action)
        # print(f"t={t}, action={action}")

    hist["bruteforce"] = h

    # Simulate Leitner
    h = np.zeros(N_ITER, dtype=int)
    np.random.seed(0)
    teacher = Leitner(n_item=N_ITEM)
    learner = ExponentialForgetting(param=PARAM, n_item=N_ITEM)

    for t in tqdm(range(N_ITER)):
        item = teacher.ask()
        r = learner.recall(item=item)
        teacher.update(item=item, response=r)
        h[t] = item

    hist["leitner"] = h

    # Simulate Threshold Teacher
    h = np.zeros(N_ITER, dtype=int)
    np.random.seed(0)
    teacher = ThresholdTeacher(n_item=N_ITEM,
                               learnt_threshold=THR)

    for t in tqdm(range(N_ITER)):
        item = teacher.ask(param=PARAM)
        h[t] = item

    hist["threshold"] = h

    data = DataFigSingle(learner_model=ExponentialForgetting,
                         param=PARAM,
                         param_labels=("alpha", "beta"),
                         n_session=1)

    condition_labels = hist.keys()

    for i, cd in enumerate(condition_labels):

        d = ExponentialForgetting().stats_ex_post(
            param=PARAM, hist=hist[cd],
            learnt_thr=THR,
        )

        data.add(
            n_learnt=d.n_learnt,
            n_seen=d.n_seen,
            p_item=d.p_item,
            label=cd,
            post_mean=d.post_mean,
            post_std=d.post_std
        )

    fig_single(data=data, fig_folder=FIG_FOLDER, time_scale=1)


if __name__ == "__main__":
    main()
