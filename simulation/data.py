import numpy as np
from tqdm import tqdm


class SimulatedData:

    def __init__(self, model, param, tk, verbose=False):

        self.n_item = tk.n_item
        t_max = tk.t_max

        agent = model(param=param, tk=tk, verbose=verbose)

        self.questions = np.asarray([list(tk.kanji).index(q) for q in
                                     tk.question_list], dtype=int)
        self.replies = np.zeros(t_max, dtype=int)
        self.success = np.zeros(t_max, dtype=int)
        self.possible_replies = np.zeros((t_max, tk.n_possible_replies),
                                         dtype=int)

        if verbose:
            print(f"Generating data with model {model.__name__}...")
            gen = tqdm(range(t_max))
        else:
            gen = range(t_max)

        for t in gen:
            q = self.questions[t]
            for i in range(tk.n_possible_replies):
                self.possible_replies[t, i] =\
                    list(tk.meaning).index(tk.possible_replies_list[t][i])

            r = agent.decide(question=q,
                             possible_replies=self.possible_replies[t])
            agent.learn(question=q)

            self.replies[t] = r
            self.success[t] = q == r

        self.times = [None for _ in range(t_max)]
