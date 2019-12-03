import numpy as np

from fit.pygpgo.classic import PYGPGOFit


class Psychologist:

    def __init__(self,
                 student_model,
                 task_param,
                 n_jobs, init_eval, max_iter, timeout,
                 testing_period, exploration_ratio):

        self.task_param = task_param
        self.testing_period = testing_period
        self.exploration_ratio = exploration_ratio

        self.student_model = student_model

        self.opt = PYGPGOFit(
            verbose=False,
            n_jobs=n_jobs,
            init_evals=init_eval, max_iter=max_iter,
            timeout=timeout)

    @staticmethod
    def most_informative(
            n_item,
            n_iteration,
            task_param,
            student_model, eval_param, hist_item,
            t
    ):

        n_param_set = len(eval_param)

        p_recall = np.zeros((n_item, n_param_set))
        for j in range(n_param_set):

            agent = student_model(param=eval_param[j],
                                  n_iteration=n_iteration,
                                  hist=hist_item, t=t,
                                  **task_param)

            for i in range(n_item):
                p_recall[i, j] = agent.p(i)

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

        return exploration

    def update_estimates(self, hist_item, hist_success, t):

        return self.opt.evaluate(
            t=t,
            task_param=self.task_param,
            hist_item=hist_item,
            hist_success=hist_success,
            model=self.student_model, )


