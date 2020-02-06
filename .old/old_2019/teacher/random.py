import numpy as np

from teacher.metaclass import GenericTeacher


class RandomTeacher(GenericTeacher):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_next_node(self, **kwargs):
        return np.random.randint(self.n_item)
