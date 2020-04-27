import numpy as np
from tqdm import tqdm


class Teacher:

    def __init__(self, n_item, n_iter_per_ss, n_iter_between_ss):

        self.n_pres = np.zeros(n_item, dtype=int)
        self.delta = np.zeros(n_item, dtype=int)

        self.n_iter_per_ss = n_iter_per_ss
        self.n_iter_between_ss = n_iter_between_ss

        self.items = np.arange(n_item)
        self.n_item = n_item

    def ask(self):

        item = np.random.choice(self.items)

        return item

    def teach(self, n_iter, seed=0, **kwargs):

        np.random.seed(seed)
        h = np.zeros(n_iter, dtype=int)

        for t in tqdm(range(n_iter)):
            item = self.ask()
            h[t] = item
        return h
