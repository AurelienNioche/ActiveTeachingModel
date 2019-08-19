import numpy as np
from tqdm import tqdm


class Data:

    def __init__(self, n_item, questions, replies, times=None,
                 possible_replies=None):

        self.n_item = n_item

        self.questions = questions
        self.replies = replies

        self.success = self.questions[:] == self.replies[:]

        self.t_max = len(self.questions)

        if times is None:
            self.times = [None for _ in range(self.t_max)]
        else:
            self.times = times

        if possible_replies is not None:
            self.possible_replies = possible_replies


class SimulatedData(Data):

    def __init__(self, model, param, tk, verbose=False):

        n_item = tk.n_item
        t_max = tk.t_max

        agent = model(param=param, tk=tk, verbose=verbose)

        questions = np.asarray([list(tk.kanji).index(q) for q in
                                tk.question_list], dtype=int)
        replies = np.zeros(t_max, dtype=int)
        possible_replies = np.zeros((t_max, tk.n_possible_replies),
                                    dtype=int)

        if verbose:
            print(f"Generating data with model {model.__name__}...")
            gen = tqdm(range(t_max))
        else:
            gen = range(t_max)

        for t in gen:
            q = questions[t]
            for i in range(tk.n_possible_replies):
                possible_replies[t, i] =\
                    list(tk.meaning).index(tk.possible_replies_list[t][i])

            r = agent.decide(item=q,
                             possible_replies=possible_replies[t])
            agent.learn(item=q)

            replies[t] = r

        times = [None for _ in range(t_max)]

        super().__init__(
            n_item=n_item,
            questions=questions,
            replies=replies,
            times=times,
            possible_replies=possible_replies

        )
