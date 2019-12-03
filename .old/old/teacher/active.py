# import copy

import numpy as np
# import itertools as it

from teacher.metaclass import GenericTeacher


class Active(GenericTeacher):

    version = 4

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v, ord=1)
        if norm == 0:
            return v
        return v / norm

    def _compute_usefulness(
            self,
            t,
            n_iteration,
            hist_item,
            task_param,
            student_param,
            student_model):
        """

        Calculate Usefulness of items
        """

        agent = student_model(
            param=student_param,
            n_iteration=n_iteration,
            hist=hist_item, t=t,
            **task_param)

        usefulness = np.zeros(self.n_item)

        for i in range(self.n_item):
            u_i = 0
            agent.learn(i)
            for j in range(self.n_item):
                next_p_recall_j_after_i = agent.p_recall(j)
                u_i += next_p_recall_j_after_i ** 2
            agent.unlearn()
            usefulness[i] = u_i

        return usefulness

    def _get_next_node(
            self,
            t,
            n_iteration,
            hist_success,
            hist_item,
            task_param,
            student_param,
            student_model):

        usefulness = self._compute_usefulness(
            t=t,
            n_iteration=n_iteration,
            hist_item=hist_item,
            student_param=student_param,
            student_model=student_model,
            task_param=task_param
        )

        candidates = np.where(usefulness == np.max(usefulness))[0]
        item = np.random.choice(candidates)
        return item


