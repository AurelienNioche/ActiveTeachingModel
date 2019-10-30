import numpy as np

from . revised import AdaptiveRevised


class Teacher(AdaptiveRevised):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.learning_value = np.zeros(len(self.possible_design))

    def _update_learning_value(self):

        for i, x in enumerate(self.possible_design):
            for j, param in enumerate(self.grid_param):

                learner = self.learner_model(
                    t=len(self.hist),
                    hist=self.hist,
                    param=param)

                learner.learn(item=x)

                v = np.zeros(len(self.possible_design))

                for l, y in enumerate(self.possible_design):

                    v[l] = self.log_lik_bernoulli(1, learner.p_recall(item=y))

                self.learning_value[i] = self.log_post[i] * np.sum(v)

    def get_design(self, kind='optimal'):
        # type: (str) -> int
        r"""
        Choose a design with a given type.

        * ``optimal``: an optimal design :math:`d^*` that maximizes the mutual
          information.
        * ``random``: a design randomly chosen.

        Parameters
        ----------
        kind : {'optimal', 'random'}, optional
            Type of a design to choose

        Returns
        -------
        design : int
            A chosen design
        """

        self._compute_log_lik()

        if kind == 'optimal':
            self._update_mutual_info()
            design = self._select_design(self.mutual_info)

        elif kind == 'random':
            design = np.random.choice(self.possible_design)

        elif kind == 'pure_teaching':
            self._update_learning_value()
            design = self._select_design(self.learning_value)

        elif kind == 'adaptive_teaching':
            if np.max(self.post_sd) > 0.1:
                self._update_mutual_info()
                design = self._select_design(self.mutual_info)
            else:
                self._update_learning_value()
                design = self._select_design(self.learning_value)

        else:
            raise ValueError(
                'The argument kind should be "optimal" or "random".')
        return design