import numpy as np
from learner.act_r import ActR


class ActRTracking(ActR):

    def __init__(self,  task_features, parameters=None, verbose=False):

        super().__init__(task_features=task_features, parameters=parameters, verbose=verbose)

        self.p = np.zeros((self.tk.t_max, self.tk.n_items))
        self.a = np.zeros((self.tk.t_max, self.tk.n_items))

    def learn(self, question):

        super().learn(question)

        for i in range(self.tk.n_items):

            self.p[self.t-1, i] = self.p_recall(i)
            self.a[self.t-1, i] = self._activation_function(i)
