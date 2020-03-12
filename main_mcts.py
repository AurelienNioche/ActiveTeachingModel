import os

import numpy as np

from mcts.mcts import MCTS
from mcts.state import State

from model.teacher.leitner import Leitner
from model.learner.learner import ExponentialForgetting

from plot.single import DataFigSingle, fig_single

import itertools as it

from tqdm import tqdm


FIG_FOLDER = os.path.join("fig", os.path.basename(__file__).split(".")[0])
os.makedirs(FIG_FOLDER, exist_ok=True)

N_ITEM = 10
PARAM = (0.02, 0.2)
THR = 0.9
N_ITER = 200

HORIZON = 10

MCTS_ITERATION_LIMIT = 10000


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


class BasicLearnerState(State):

    def __init__(self, seen):

        self.seen = seen

    def get_possible_actions(self):
        """Returns an iterable of all actions which can be taken
        from this state"""
        return np.arange(N_ITEM)

    def take_action(self, action):
        """Returns the state which results from taking action 'action'"""
        new_seen = self.seen.copy()
        new_seen[action] = 1
        return self.__class__(new_seen)

    def is_terminal(self):
        """Returns whether this state is a terminal state"""
        return False

    def get_reward(self):
        """"Returns the reward for this state"""
        return np.sum(self.seen == 1)

    def __str__(self):
        return f"State: {self.seen}"


class BruteForceTeacher:

    def __init__(self, horizon=2):
        self.n_pres = np.zeros(N_ITEM, dtype=int)
        self.delta = np.zeros(N_ITEM, dtype=int)

        self.horizon = horizon

        self.t = 0

    def ask(self):

        p = []

        n_item = len(self.n_pres)
        seen = self.n_pres[:] > 0
        unseen = np.logical_not(seen)
        n_seen = np.sum(seen)

        items = np.arange(n_item)

        nodes = []

        for t in range(self.horizon):
            nodes.append([])

            for item in range(N_ITEM):
                nodes[t].append(item)


        if n_seen == 0:
            item = 0

        else:

            seen = self.n_pres[:] > 0

            fr = PARAM[0] * (1 - PARAM[1]) ** (self.n_pres[seen] - 1)

            p = np.exp(-fr * self.delta[seen])

            min_p = np.min(p)

            if n_seen == n_item or min_p <= THR:
                item = np.random.choice(items[seen][p[:] == min_p])

            else:
                item = items[unseen][0]

        self.n_pres[item] += 1

        # Increment delta for all items
        self.delta[:] += 1
        # ...except the one for the selected design that equal one
        self.delta[item] = 1

        return item


class LearnerState(State):

    def __init__(self, n_pres, delta, t, cumulative_reward, first=False):

        self.n_pres = n_pres
        self.delta = delta

        self.t = t

        self._instant_reward = None
        self._possible_actions = None

        if first is not True:
            self.cumulative_reward = cumulative_reward \
                                     + self.get_instant_reward()
        else:
            self.cumulative_reward = - self.get_instant_reward()

    def get_possible_actions(self):
        """Returns an iterable of all actions which can be taken
        from this state"""
        if self._possible_actions is None:
            seen = self.n_pres[:] > 0
            n_seen = np.sum(seen)
            if n_seen == 0:
                self._possible_actions = np.arange(1)
            elif n_seen == N_ITEM:
                self._possible_actions = np.arange(N_ITEM)
            else:
                already_seen = np.arange(N_ITEM)[seen]
                new = np.max(already_seen)+1
                self._possible_actions = list(already_seen) + [new]
        return self._possible_actions

        # return np.arange(N_ITEM)

    def take_action(self, action, fake=True):
        """Returns the state which results from taking action 'action'"""
        n_pres = self.n_pres.copy()
        delta = self.delta.copy()

        n_pres[action] += 1

        # Increment delta for all items
        delta[:] += 1
        # ...except the one for the selected design that equal one
        delta[action] = 1
        t = self.t+1

        if fake:
            new_state = self.__class__(n_pres=n_pres, delta=delta, t=t,
                                       cumulative_reward=self.cumulative_reward)
        else:
            new_state = self.__class__(n_pres=n_pres, delta=delta,
                                       t=0, cumulative_reward=0, first=True)

        return new_state

    def is_terminal(self):
        """Returns whether this state is a terminal state"""
        return self.t >= HORIZON

    def get_instant_reward(self):
        """"Returns the INSTANT reward for this state"""

        if self._instant_reward is not None:
            return self._instant_reward
        else:
            seen = self.n_pres[:] > 0

            fr = PARAM[0] * (1 - PARAM[1]) ** (self.n_pres[seen] - 1)

            p = np.exp(-fr * self.delta[seen])
            self._instant_reward = np.sum(p > THR)
            return self._instant_reward

    def get_reward(self):
        """"Returns the CUMULATIVE reward for this state"""
        return self.cumulative_reward / HORIZON
    # def __str__(self):
    #     return f"State: "


def main():

    leaner_state = LearnerState(n_pres=np.zeros(N_ITEM, dtype=bool),
                                delta=np.zeros(N_ITEM, dtype=int), t=0,
                                cumulative_reward=0, first=True)
    hist = {}

    h = np.zeros(N_ITER, dtype=int)
    np.random.seed(0)
    for t in tqdm(range(N_ITER)):
        m = MCTS(iteration_limit=MCTS_ITERATION_LIMIT, verbose=False)
        action = m.run(initial_state=leaner_state)

        leaner_state = leaner_state.take_action(action, fake=False)

        h[t] = action
        # print(f"t={t}, action={action}")

    hist["mcts"] = h

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
