import numpy as np

from teacher.metaclass import GenericTeacher


class RandomTeacher(GenericTeacher):

    def __init__(self, n_item=20, t_max=100, grades=(1, ), seed=123,
                 handle_similarities=True, normalize_similarity=False,
                 verbose=False):

        super().__init__(n_item=n_item, t_max=t_max, grades=grades, seed=seed,
                         normalize_similarity=normalize_similarity,
                         handle_similarities=handle_similarities,
                         verbose=verbose)

        np.random.seed(seed)

    def _get_next_node(self, agent=None):

        question = np.random.randint(self.tk.n_item)
        return question
