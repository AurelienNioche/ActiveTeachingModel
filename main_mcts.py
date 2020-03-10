import os

import numpy as np

from mcts.mcts import MCTS
from mcts.state import State

from model.teacher.leitner import Leitner
from model.learner.learner import ExponentialForgetting

from plot.single import DataFigSingle, fig_single

from tqdm import tqdm


FIG_FOLDER = os.path.join("fig", os.path.basename(__file__).split(".")[0])
os.makedirs(FIG_FOLDER, exist_ok=True)

N_ITEM = 100
PARAM = (0.02, 0.2)
THR = 0.9
N_ITER = 100

HORIZON = 5

MCTS_ITERATION_LIMIT = 1000


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


class LearnerState(State):

    def __init__(self, n_pres, delta, t):

        self.n_pres = n_pres
        self.delta = delta

        self.t = t

        self._reward = None
        self._possible_actions = None

    def get_possible_actions(self):
        """Returns an iterable of all actions which can be taken
        from this state"""
        if self._possible_actions is None:
            seen = self.n_pres[:] > 0
            n_seen = np.sum(seen)
            if n_seen == 0:
                return np.arange(1)
            elif n_seen == N_ITEM:
                self._possible_actions = np.arange(N_ITEM)
            else:
                already_seen = np.arange(N_ITEM)[seen]
                new = np.max(already_seen)+1
                self._possible_actions = list(already_seen) + [new]
        return self._possible_actions

        # return np.arange(N_ITEM)

    def take_action(self, action):
        """Returns the state which results from taking action 'action'"""
        n_pres = self.n_pres.copy()
        delta = self.delta.copy()

        n_pres[action] += 1

        # Increment delta for all items
        delta[:] += 1
        # ...except the one for the selected design that equal one
        delta[action] = 1
        t = self.t+1
        return self.__class__(n_pres=n_pres, delta=delta, t=t)

    def is_terminal(self):
        """Returns whether this state is a terminal state"""
        return self.t >= HORIZON

    def get_reward(self):
        """"Returns the reward for this state"""

        if self._reward is not None:
            return self._reward
        else:
            seen = self.n_pres[:] > 0

            fr = PARAM[0] * (1 - PARAM[1]) ** (self.n_pres[seen] - 1)

            p = np.exp(-fr * self.delta[seen])
            self._reward = np.sum(p > THR)
            return self._reward

    # def __str__(self):
    #     return f"State: "


def main():

    leaner_state = LearnerState(n_pres=np.zeros(N_ITEM, dtype=bool),
                                delta=np.zeros(N_ITEM, dtype=int), t=0)
    hist = {}

    h = np.zeros(N_ITER, dtype=int)
    np.random.seed(0)
    for t in tqdm(range(N_ITER)):
        m = MCTS(iteration_limit=MCTS_ITERATION_LIMIT, verbose=False)
        action = m.run(initial_state=leaner_state)

        leaner_state = leaner_state.take_action(action)
        leaner_state.t = 0

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
