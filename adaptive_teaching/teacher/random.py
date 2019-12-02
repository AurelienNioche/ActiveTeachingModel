import numpy as np

from . metaclass import GenericTeacher


class RandomTeacher(GenericTeacher):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def ask(self, best_param):
        return np.random.randint(self.n_item)
