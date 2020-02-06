import numpy as np
from tqdm import tqdm


class Data:

    def __init__(self, n_item, questions, replies, times=None,
                 possible_replies=None):

        self.n_item = n_item

        self.questions = questions
        self.replies = replies

        self.success = self.questions[:] == self.replies[:]

        self.n_iteration = len(self.questions)

        if times is None:
            self.times = [None for _ in range(self.n_iteration)]
        else:
            self.times = times

        if possible_replies is not None:
            self.possible_replies = possible_replies


class SimulatedData(Data):

    def __init__(self, model, param, tk, verbose=False):

        n_item = tk.n_item
        n_iteration = tk.n_iteration

        agent = model(param=param, tk=tk, verbose=verbose)

        questions = np.asarray([list(tk.kanji).index(q) for q in
                                tk.question_list], dtype=int)
        replies = np.zeros(n_iteration, dtype=int)
        possible_replies = np.zeros((n_iteration, tk.n_possible_replies),
                                    dtype=int)

        if verbose:
            print(f"Generating data with model {model.__name__}...")
            gen = tqdm(range(n_iteration))
        else:
            gen = range(n_iteration)

        for t in gen:
            q = questions[t]
            for i in range(tk.n_possible_replies):
                possible_replies[t, i] =\
                    list(tk.meaning).index(tk.possible_replies_list[t][i])

            r = agent.decide(item=q,
                             possible_replies=possible_replies[t])
            agent.learn(item=q)

            replies[t] = r

        times = [None for _ in range(n_iteration)]

        super().__init__(
            n_item=n_item,
            questions=questions,
            replies=replies,
            times=times,
            possible_replies=possible_replies

        )
