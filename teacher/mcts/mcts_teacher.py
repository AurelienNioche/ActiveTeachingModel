import numpy as np
from tqdm import tqdm
from copy import deepcopy

from learner.learner import Learner
from psychologist.psychologist import Psychologist

from . mcts.mcts import MCTS
from . mcts.state import LearnerState
from . mcts.reward import Reward
from . mcts.rollout import RolloutRandom, RolloutThreshold


class DynamicParam:

    def __init__(self):
        self.t = 0
        self.c_iter = None
        self.terminal_t = None


class MCTSTeacher:

    def __init__(self,
                 learner,
                 reward,
                 terminal_t,
                 n_item,
                 thr,
                 horizon=20,
                 iter_limit=500,
                 fixed_window=False,
                 rollout='threshold'):

        self.iter_limit = iter_limit
        self.fixed_window = fixed_window

        self.dyn_param = DynamicParam()
        self.terminal_t = terminal_t

        self.horizon = horizon

        if rollout == 'threshold':
            rollout = RolloutThreshold(n_item=n_item, tau=thr)
        elif rollout == 'random':
            rollout = RolloutRandom(n_item=n_item, tau=thr)
        else:
            raise ValueError

        self.learner_state = \
            LearnerState(
                learner=learner,
                rollout=rollout,
                horizon=horizon,
                reward=reward,
                dyn_param=self.dyn_param,
            )

        self.c_iter = 0

    def _revise_goal(self):

        self.dyn_param.c_iter = self.c_iter

        if not self.fixed_window:
            self.dyn_param.terminal_t = self.terminal_t
        else:
            if self.dyn_param.terminal_t is None \
                    or self.dyn_param.terminal_t == self.learner_state.learner.t:
                if self.dyn_param.terminal_t is None:
                    self.dyn_param.terminal_t = 0

                c_iter_ss = self.learner_state.learner.c_iter_ss
                n_breaks = (c_iter_ss + self.horizon) // self.learner_state.learner.n_iter_per_ss
                to_add = self.horizon + n_breaks * self.learner_state.learner.n_iter_between_ss
                self.dyn_param.terminal_t += to_add
            else:
                pass

    def ask(self):

        m = MCTS(iteration_limit=self.iter_limit)

        self._revise_goal()

        self.learner_state.reset()
        best_action = m.run(initial_state=self.learner_state)

        return best_action

    def update(self, item):

        self.learner_state = self.learner_state.take_action(item)
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
        learner = Learner.get(tk)
        teacher = cls(
            learner=learner,
            reward=reward,
            terminal_t=tk.terminal_t,
            n_item=tk.n_item,
            thr=tk.thr,
            **tk.mcts_kwargs)
        return teacher.teach(n_iter=tk.n_iter, seed=tk.seed)


class MCTSPsychologist(MCTSTeacher):

    def __init__(self,
                 psychologist,
                 learner,
                 reward,
                 terminal_t,
                 n_item,
                 thr,
                 horizon=20,
                 iter_limit=500,
                 fixed_window=False,
                 rollout='threshold'):

        self.learner = learner
        self.psychologist = psychologist

        super().__init__(
            n_item=n_item,
            thr=thr,
            horizon=horizon,
            iter_limit=iter_limit,
            fixed_window=fixed_window,
            rollout=rollout,
            learner=deepcopy(learner),
            reward=reward,
            terminal_t=terminal_t)

    def ask(self):

        m = MCTS(iteration_limit=self.iter_limit)

        self._revise_goal()

        self.learner_state.reset()
        self.learner_state.learner.param = self.psychologist.post_mean

        item = m.run(initial_state=self.learner_state)
        return item

    def update(self, item):

        self.learner_state = self.learner_state.take_action(item)
        response = self.learner.reply(item)
        self.psychologist.update(item=item, response=response)
        self.learner.update(item)
        self.c_iter += 1

    @classmethod
    def run(cls, tk):
        learner = Learner.get(tk)
        psychologist = Psychologist.get(n_iter=tk.n_iter, learner=learner)
        reward = Reward(n_item=tk.n_item, tau=tk.thr)
        teacher = cls(
            psychologist=psychologist,
            learner=learner,
            reward=reward,
            terminal_t=tk.terminal_t,
            n_item=tk.n_item,
            thr=tk.thr,
            **tk.mcts_kwargs)
        hist = teacher.teach(n_iter=tk.n_iter, seed=tk.seed)
        if psychologist.hist_pm is not None \
                and psychologist.hist_psd is not None:
            param_recovery = {
                'post_mean': psychologist.hist_pm,
                'post_std': psychologist.hist_psd}
            return hist, param_recovery
        else:
            return hist
