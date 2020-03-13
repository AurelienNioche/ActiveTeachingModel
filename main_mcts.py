import os

import numpy as np

from mcts.mcts import MCTS
from mcts.state import LearnerState

from model.teacher.leitner import Leitner
from model.learner.learner import ExponentialForgetting

from plot.single import DataFigSingle, fig_single

from tqdm import tqdm


FIG_FOLDER = os.path.join("fig", os.path.basename(__file__).split(".")[0])
os.makedirs(FIG_FOLDER, exist_ok=True)

N_ITEM = 4
PARAM = (0.02, 0.2)
THR = 0.9
N_ITER = 10

MCTS_HORIZON = 10

MCTS_ITERATION_LIMIT = 10000


class BruteForceTeacher:

    def __init__(self, horizon=2):
        self.n_pres = np.zeros(N_ITEM, dtype=int)
        self.delta = np.zeros(N_ITEM, dtype=int)

        self.horizon = horizon

        self.t = 0

    def ask(self, learner_state):

        value = []
        root_actions = []

        for t in range(self.horizon):

            print("t", t)

            actions = learner_state.get_possible_actions()
            print("possible actions", actions)

            for a in actions:
                print("evaluate action", a)
                learner_state = learner_state.take_action(
                    action=a, root=t == 0)
                if t == self.horizon-1:
                    value.append(learner_state.get_reward())
                    root_actions.append(learner_state.root_action)

        best_idx = int(np.argmax(value))
        best_action = root_actions[best_idx]
        print("value", value)
        print("best idx", best_idx)
        print("best action", best_action)
        return best_action


class ThresholdTeacher:

    def __init__(self):

        self.n_pres = np.zeros(N_ITEM, dtype=int)
        self.delta = np.zeros(N_ITEM, dtype=int)

        self.t = 0

    def ask(self):

        n_item = len(self.n_pres)
        seen = self.n_pres[:] > 0
        n_seen = np.sum(seen)

        items = np.arange(n_item)

        if n_seen == 0:
            item = 0

        else:
            fr = PARAM[0] * (1 - PARAM[1]) ** (self.n_pres[seen] - 1)
            p = np.exp(-fr * self.delta[seen])

            min_p = np.min(p)

            if n_seen == n_item or min_p <= THR:
                item = np.random.choice(items[seen][p[:] == min_p])

            else:
                unseen = np.logical_not(seen)
                item = items[unseen][0]

        self.n_pres[item] += 1

        # Increment delta for all items
        self.delta[:] += 1
        # ...except the one for the selected design that equal one
        self.delta[item] = 1

        return item


def main():

    # leaner_state = LearnerState(n_pres=np.zeros(N_ITEM, dtype=bool),
    #                             delta=np.zeros(N_ITEM, dtype=int),
    #                             learnt_thr=THR,
    #                             horizon=HORIZON,
    #                             param=PARAM)
    # leaner_state.t = 0
    # leaner_state.cumulative_reward = - leaner_state.get_instant_reward()
    #
    # hist = {}
    #
    # h = np.zeros(N_ITER, dtype=int)
    # np.random.seed(0)
    # for t in tqdm(range(N_ITER)):
    #     m = MCTS(iteration_limit=MCTS_ITERATION_LIMIT, verbose=False)
    #     action = m.run(initial_state=leaner_state)
    #
    #     leaner_state = leaner_state.take_action(action, fake=False)
    #
    #     h[t] = action
    #     # print(f"t={t}, action={action}")
    #
    # hist["mcts"] = h

    # Simulate bruteforce
    leaner_state = LearnerState(n_pres=np.zeros(N_ITEM, dtype=bool),
                                delta=np.zeros(N_ITEM, dtype=int),
                                learnt_thr=THR,
                                horizon=MCTS_HORIZON,
                                param=PARAM)
    leaner_state.t = 0
    leaner_state.cumulative_reward = - leaner_state.get_instant_reward()

    hist = {}

    h = np.zeros(N_ITER, dtype=int)
    np.random.seed(0)
    perfect_teacher = BruteForceTeacher()
    for t in tqdm(range(N_ITER)):

        action = perfect_teacher.ask(learner_state=leaner_state)
        print("teacher choose", action)
        leaner_state = leaner_state.take_action(action, fake=False)

        h[t] = action
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
    teacher = ThresholdTeacher()

    for t in tqdm(range(N_ITER)):
        item = teacher.ask()
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
