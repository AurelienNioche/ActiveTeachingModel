import numpy as np

from fit.pygpgo.classic import PYGPGOFit


class Psychologist:

    def __init__(self,
                 student_model, n_item, n_iteration,
                 n_jobs, init_eval, max_iter, timeout,
                 testing_period, exploration_ratio):

        self.testing_period = testing_period
        self.exploration_ratio = exploration_ratio

        self.student_model = student_model

        self.opt = PYGPGOFit(
            verbose=False,
            n_jobs=n_jobs,
            init_evals=init_eval, max_iter=max_iter,
            timeout=timeout)

        self.model_learner = student_model(
            n_item=n_item,
            n_iteration=n_iteration,
            param=student_model.generate_random_parameters()
        )

    @staticmethod
    def most_informative(tk, student_model, eval_param,
                         t_max, questions):

        n_param_set = len(eval_param)

        p_recall = np.zeros((tk.n_item, n_param_set))
        for j in range(n_param_set):

            agent = student_model(param=eval_param[j], tk=tk)

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

    def is_time_for_exploration(self, t):

        if t == 0:
            exploration = False

        elif t < self.testing_period:
            exploration = t % 2 == 1

        else:
            exploration = np.random.random() <= self.exploration_ratio
            print('yeah')

        return exploration

    def update_estimates(self):

        self.opt.evaluate(
            data=data_view, model=self.student_model, )

        self.model_learner.set_parameters(self.opt.best_param.copy())


