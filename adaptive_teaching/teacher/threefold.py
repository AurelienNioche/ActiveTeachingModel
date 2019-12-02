from . metaclass import GenericTeacher


class Threefold(GenericTeacher):

    def __int__(self, task_param, learner_model):

        super().__init__(task_param=task_param, learner_model=learner_model)

    def ask(self, best_param):

        pass

    # def _update_learning_value(self):
    #
    #     for i in range(self.n_design):
    #         self.log_lik[i, :, :] = self._log_p(i)
    #
    #         self._update_history(i)
    #
    #         v = np.zeros(self.n_design)
    #
    #         for j in range(self.n_design):
    #
    #             v[j] = logsumexp(self.log_post[:] + self._log_p(j)[:, 1])
    #
    #         self.learning_value[i] = np.sum(v)
    #
    #         self._cancel_last_update_history()
