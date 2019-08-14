import numpy as np


class Psychologist:

    @staticmethod
    def most_informative(tk, student_model, param_set,
                         t_max, questions):

        n_param_set = len(param_set)

        p_recall = np.zeros((tk.n_item, n_param_set))
        for j in range(n_param_set):

            agent = student_model(param=param_set[j], tk=tk)

            for t in range(t_max):
                agent.learn(questions[t])

            for i in range(tk.n_item):
                p_recall[i, j] = agent.p_recall(i)

        std_per_item = np.std(p_recall, axis=1)

        max_std = np.max(std_per_item)
        if max_std == 0:
            non_seen_item = np.where(p_recall == 0)[0]
            return np.random.choice(non_seen_item)

        else:
            max_std_items = np.where(std_per_item == max_std)[0]
            return np.random.choice(max_std_items)


