import numpy as np

from fit.gpyopt import Gpyopt

from teacher.metaclass import GenericTeacher


class SimplePsychologist(GenericTeacher):

    version = 0.1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_next_node(
            self,
            student_model,
            hist_item,
            hist_success,
            n_iteration,
            task_param,
            t,
            opt_param=None,
            **kwargs
    ):

        if t == 0:
            item = np.random.randint(self.n_item)

        else:
            n_param_set = 1000

            p_recall = np.zeros((self.n_item, n_param_set))
            for j in range(n_param_set):

                # param = self.hist_param[j]

                # if 'default_param' in task_param:
                #     param = task_param['default_param']
                #     param.update(student_model.generate_random_parameters())
                #
                # else:
                param = student_model.generate_random_parameters()

                agent = student_model(
                    param=param,
                    n_iteration=n_iteration,
                    hist=hist_item.copy(),
                    t=t,
                    **task_param)

                for i in range(self.n_item):
                    agent.learn(i)
                    p_recall[i, j] = agent.p_recall(i)
                    agent.unlearn()

            std_per_item = np.std(p_recall, axis=1)

            max_std = np.max(std_per_item)
            if max_std == 0:
                non_seen_item = np.where(p_recall == 0)[0]
                item = np.random.choice(non_seen_item)

            else:
                max_std_items = np.where(std_per_item == max_std)[0]
                item = np.random.choice(max_std_items)

        return item


class Psychologist(SimplePsychologist):

    def __init__(self,
                 fit_class=None,
                 fit_param=None,
                 testing_period=None, exploration_ratio=None,
                 exploration_threshold=None,
                 **kwargs):

        super().__init__(**kwargs)

        if fit_class is None:
            self.fit_class = Gpyopt
        else:
            self.fit_class = fit_class

        if fit_param is None:
            self.fit_param = {}
        else:
            self.fit_param = fit_param

        self.testing_period = testing_period
        self.exploration_ratio = exploration_ratio
        self.exploration_threshold = exploration_threshold

        self.hist_item = None
        self.hist_success = None
        self.t = None

        self.hist_values = None
        self.hist_param = None

        self.hist_best_values = []
        self.hist_best_param = []

        self.best_param = None
        self.best_value = None
        self.estimated_var = None

        self.student_model = None
        self.task_param = None

    def _get_next_node(
            self,
            student_model,
            hist_item,
            hist_success,
            n_iteration,
            task_param,
            t,
            **kwargs
    ):
        self.student_model = student_model
        self.task_param = task_param

        self.hist_item = hist_item
        self.hist_success = hist_success

        if t == 0:
            item = np.random.randint(self.n_item)
        else:
            if self.hist_param is None:
                self.update_estimates(
                    student_model=student_model,
                    task_param=task_param,
                    hist_item=hist_item,
                    hist_success=hist_success,
                    t=t
                )

            # noinspection PyTypeChecker
            # n_param_set = len(self.hist_param)

            n_param_set = 100

            p_recall = np.zeros((self.n_item, n_param_set))
            for j in range(n_param_set):

                # param = self.hist_param[j]

                param = self.student_model.generate_random_parameters()

                agent = self.student_model(
                    param=param,
                    n_iteration=n_iteration,
                    hist=self.hist_item.copy(),
                    t=t,
                    **task_param)

                for i in range(self.n_item):
                    agent.learn(i)
                    p_recall[i, j] = agent.p_recall(i)
                    agent.unlearn()

            std_per_item = np.std(p_recall, axis=1)

            max_std = np.max(std_per_item)
            if max_std == 0:
                non_seen_item = np.where(p_recall == 0)[0]
                item = np.random.choice(non_seen_item)

            else:
                max_std_items = np.where(std_per_item == max_std)[0]
                item = np.random.choice(max_std_items)

        self.hist_param = None
        return item

    def is_time_for_exploration(self, t):

        if t == 0:
            return False
        elif t == 1:
            return True
        elif t == 2:
            return False

        assert self.exploration_ratio is None \
            or self.exploration_threshold is not None, \
            "Choose one or the other!"

        assert not (self.exploration_threshold is not None
                    and self.fit_class != Gpyopt), "Not expected!"

        if self.exploration_ratio is not None:

            if t < self.testing_period:
                exploration = t % 2 == 1

            else:
                exploration = np.random.random() <= self.exploration_ratio

        else:
            exploration = self.estimated_var >= self.exploration_threshold

        return exploration

    def update_estimates(self,
                         student_model,
                         task_param,
                         hist_item,
                         hist_success,
                         t):

        self.student_model = student_model
        self.task_param = task_param

        self.hist_values = []
        self.hist_param = []

        self.hist_item = hist_item
        self.hist_success = hist_success
        self.t = t

        f = self.fit_class(model=student_model)
        f.evaluate(
            hist_question=hist_item, hist_success=hist_success,
            task_param=task_param, **self.fit_param
        )

        self.best_param = f.best_param
        self.best_value = f.best_value
        self.hist_values = f.obj_values
        self.hist_param = f.history_eval_param
        self.hist_best_param.append(self.best_param)
        self.hist_best_values.append(self.best_value)

        if self.fit_class == Gpyopt:
            self.estimated_var = f.get_estimated_var()

        # print("BEST PARAM", self.best_param)

        return self.best_param
