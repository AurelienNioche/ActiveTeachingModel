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

N_ITEM = 100
PARAM = (0.02, 0.2)
THR = 0.9
N_ITER = 1000

MCTS_HORIZON = 10

MCTS_ITERATION_LIMIT = 10000


class BruteForceTeacher:

    def __init__(self, horizon=2, verbose=0):

        self.t = 0

        self.learner_state = \
            LearnerState(
                n_pres=np.zeros(N_ITEM, dtype=int),
                delta=np.zeros(N_ITEM, dtype=int),
                learnt_thr=THR,
                horizon=horizon,
                param=PARAM)

        self.verbose = verbose

    def ask(self):
        if self.verbose:
            print("*" * 40)
            print(f"T = {self.t}")
            print("*" * 40)
            print()

        self.learner_state.reset()

        done = False

        value = []
        root_actions = []

        initial_state = self.learner_state

        learner_state = initial_state

        if self.verbose:
            actions = learner_state.get_possible_actions()
            print()
            print("Possible actions " + "*" * 10)
            print(f"t={learner_state.t}, possible actions={actions}")
            print()
            print("Evaluation" + "*" * 10)

        while True:
            actions = learner_state.get_possible_actions()

            if self.verbose==2:
                print()
                print("Possible actions " + "*" * 10)
                print(f"t={learner_state.t}, possible actions={actions}")
                print()
                print("Evaluation" + "*"*10)

            for a in actions:

                if a not in learner_state.children:
                    if self.verbose==2:
                        print(f"evaluate action {a} at t={learner_state.t}")
                    learner_state = learner_state.take_action(action=a)
                    if self.verbose==2:
                        print(f"New learner state at t={learner_state.t} "
                              f"with r={learner_state.get_instant_reward()}")
                    break

            if learner_state.is_terminal():
                if self.verbose==2:
                    print("New state is terminal.")
                value.append(learner_state.get_reward())
                root_actions.append(learner_state.action[0])

                fully_expanded = True
                while fully_expanded:
                    if self.verbose==2:
                        print("Fully expanded or terminal. Taking parent.")
                    learner_state = learner_state.parent
                    actions = learner_state.get_possible_actions()
                    fully_expanded = len(learner_state.children) == len(
                        actions)

                    root = learner_state == initial_state
                    if fully_expanded and root:
                        if self.verbose==2:
                            print("Tree is fully expanded.")
                        done = True
                        break

            if done:
                if self.verbose==2:
                    print("Ready to provide action.")
                break

        a_value = np.asarray(value)
        a_actions = np.asarray(root_actions)
        max_value = np.max(a_value)
        best_action = np.random.choice(a_actions[a_value == max_value])

        if self.verbose:
            print(f"Root action-value={[(s,v) for s, v in zip(root_actions, value)]}")
            print("Selected action", best_action)
            print("Max value", max_value)
            print("*" * 40)

        self.learner_state = self.learner_state.take_action(best_action)

        self.t += 1
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

    hist = {}

    h = np.zeros(N_ITER, dtype=int)
    np.random.seed(0)
    perfect_teacher = BruteForceTeacher()
    for t in tqdm(range(N_ITER)):

        action = perfect_teacher.ask()
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
