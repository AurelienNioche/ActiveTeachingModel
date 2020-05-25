import numpy as np
from tqdm import tqdm
from copy import deepcopy

from learner.learner import Learner
from psychologist.psychologist import Psychologist

from . mcts.mcts import MCTS
from . mcts.state import LearnerState
from . mcts.reward import Reward
from . mcts.rollout import Rollout


class DynamicParam:

    def __init__(self):
        self.t = 0
        self.c_iter = None
        self.terminal_t = None


class MCTSTeacher:

    def __init__(self,
                 learner,
                 reward,
                 rollout,
                 terminal_t,
                 horizon=20,
                 iter_limit=500,
                 fixed_window=False):

        self.iter_limit = iter_limit
        self.fixed_window = fixed_window

        self.dyn_param = DynamicParam()
        self.terminal_t = terminal_t

        self.horizon = horizon

        self.learner_state = \
            LearnerState(
                learner=learner,
                rollout=rollout,
                horizon=horizon,
                reward=reward,
                dyn_param=self.dyn_param,
            )
                # terminal_t=terminal_t)

        self.c_iter = 0

    def ask(self):

        m = MCTS(iteration_limit=self.iter_limit)

        self.dyn_param.c_iter = self.c_iter

        if not self.fixed_window:
            self.dyn_param.terminal_t = self.terminal_t
        else:
            if self.dyn_param.terminal_t is None or self.dyn_param.terminal_t == self.learner_state.learner.t:
                if self.dyn_param.terminal_t is None:
                    self.dyn_param.terminal_t = 0

                c_iter_ss = self.learner_state.learner.c_iter_ss
                n_breaks = (c_iter_ss + self.horizon) // self.learner_state.learner.n_iter_per_ss
                print("n breaks", n_breaks)
                to_add = self.horizon + n_breaks * self.learner_state.learner.n_iter_between_ss
                self.dyn_param.terminal_t += to_add
                print(f"add {to_add} to goal")
                print("New terminal t", self.dyn_param.terminal_t)
                # for i in range(self.horizon):
                #     self.dyn_param.terminal_t += 1
                #     c_iter_ss += 1
                #     if c_iter_ss >= self.learner_state.learner.n_iter_per_ss:
                #         self.dyn_param.terminal_t += self.learner_state.learner.n_iter_between_ss
                #         c_iter_ss = 0
            else:
                pass

        self.learner_state.reset()
        best_action = m.run(initial_state=self.learner_state)

        self.learner_state = self.learner_state.take_action(best_action)
        return best_action

    def update(self, item):

        self.c_iter += 1

    def teach(self, n_iter, seed=0):

        np.random.seed(seed)
        h = np.zeros(n_iter, dtype=int)

        for t in tqdm(range(n_iter)):
            item = self.ask()
            self.update(item)
            h[t] = item
        return h

    @classmethod
    def run(cls, tk):

        reward = Reward(n_item=tk.n_item, tau=tk.thr)
        rollout = Rollout(n_item=tk.n_item, tau=tk.thr)
        learner = Learner.get(tk)
        teacher = cls(
            learner=learner,
            reward=reward,
            rollout=rollout,
            terminal_t=tk.terminal_t,
            **tk.mcts_kwargs)
        return teacher.teach(n_iter=tk.n_iter, seed=tk.seed)


class MCTSPsychologist(MCTSTeacher):

    def __init__(self,
                 psychologist,
                 learner,
                 reward,
                 rollout,
                 terminal_t,
                 horizon=20,
                 iteration_limit=500):

        self.reward = reward
        self.learner = learner

        super().__init__(
            learner=deepcopy(learner),
            reward=reward,
            rollout=rollout,
            terminal_t=terminal_t,
            horizon=horizon,
            iteration_limit=iteration_limit)

        self.psychologist = psychologist

    def ask(self):

        m = MCTS(iteration_limit=self.iteration_limit)

        self.learner_state.reset()
        self.learner_state.learner.param = self.psychologist.post_mean
        self.reference_point.c_iter = self.c_iter
        item = m.run(initial_state=self.learner_state)
        return item

    def update(self, item):

        self.learner_state = self.learner_state.take_action(item)
        response = self.learner.reply(item)
        self.psychologist.update(item=item, response=response)
        self.learner.update(item)
        super().update(item)

    @classmethod
    def run(cls, tk):

        learner = Learner.get(tk)
        psychologist = Psychologist.get(n_iter=tk.n_iter, learner=learner)
        reward = Reward(n_item=tk.n_item, tau=tk.thr)
        rollout = Rollout(n_item=tk.n_item, tau=tk.thr)
        teacher = cls(
            psychologist=psychologist,
            learner=learner,
            reward=reward,
            rollout=rollout,
            terminal_t=tk.terminal_t,
            horizon=tk.mcts_horizon,
            iteration_limit=tk.mcts_iter_limit)
        hist = teacher.teach(n_iter=tk.n_iter, seed=tk.seed)
        if psychologist.hist_pm is not None \
                and psychologist.hist_psd is not None:
            param_recovery = {
                'post_mean': psychologist.hist_pm,
                'post_std': psychologist.hist_psd}
            return hist, param_recovery
        else:
            return hist
