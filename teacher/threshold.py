import numpy as np
from tqdm import tqdm

from learner.learner import Learner
from psychologist.psychologist import Psychologist


class ThresholdTeacher:

    def __init__(self, learner, learnt_threshold):

        self.learner = learner

        self.learnt_threshold = learnt_threshold

        self.items = np.arange(self.learner.n_item)

    def ask(self):

        seen = self.learner.seen
        n_seen = np.sum(seen)

        if n_seen == 0:
            items_selected = self.items

        else:
            p = self.learner.p_seen()

            min_p = np.min(p)

            if n_seen == self.learner.n_item or min_p <= self.learnt_threshold:
                items_selected = self.items[seen][p[:] == min_p]

            else:
                unseen = np.logical_not(seen)
                items_selected = self.items[unseen]

        item = np.random.choice(items_selected)
        return item

    def update(self, item):
        self.learner.update(item)

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
        learner = Learner.get(tk)
        teacher = cls(learner=learner, learnt_threshold=tk.thr)
        return teacher.teach(n_iter=tk.n_iter, seed=tk.seed)


class ThresholdPsychologist(ThresholdTeacher):

    def __init__(self, learner, learnt_threshold, psychologist):
        super().__init__(learner=learner, learnt_threshold=learnt_threshold)
        self.psychologist = psychologist
        self.true_param = self.learner.param

    def ask(self):
        self.learner.param = self.psychologist.post_mean
        return super().ask()

    def update(self, item):

        self.learner.param = self.true_param
        response = self.learner.reply(item)
        self.psychologist.update(item=item, response=response)
        self.learner.update(item)

    @classmethod
    def run(cls, tk):
        learner = Learner.get(tk)
        psychologist = Psychologist.get(n_iter=tk.n_iter, learner=learner)
        teacher = ThresholdPsychologist(
            learner=learner,
            learnt_threshold=tk.thr,
            psychologist=psychologist)
        hist = teacher.teach(n_iter=tk.n_iter, seed=tk.seed)
        if psychologist.hist_pm is not None \
                and psychologist.hist_psd is not None:
            param_recovery = {
                'post_mean': psychologist.hist_pm,
                'post_std': psychologist.hist_psd}
            return hist, param_recovery
        else:
            return hist
