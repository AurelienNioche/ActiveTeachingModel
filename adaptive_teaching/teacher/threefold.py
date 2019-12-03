from . metaclass import GenericTeacher


class Threefold(GenericTeacher):

    def __int__(self, task_param, learner_model):

        super().__init__(task_param=task_param, learner_model=learner_model)
        self.learner = learner_model(task_param=task_param)

    def ask(self, best_param):

        for i in range(self.n_item):

            # Learn new item
            self.learner.update(i)

            log_lik_t_plus_one = np.zeros((
                self.n_item,
                n_best,
                2))

            for j in range(self.n_item):
                p = self.learner.p(
                    grid_param=self.grid_param[best_param_set_idx],
                    i=i,
                )

                new_log_p = np.log(p + EPS)

                log_lik_t_plus_one[j, :, :] = new_log_p

            mutual_info_t_plus_one_for_seq_i_j = \
                self._mutual_info(log_lik_t_plus_one,
                                  self.log_post[best_param_set_idx])
            max_info_next_time_step = \
                np.max(mutual_info_t_plus_one_for_seq_i_j)

            self.mutual_info[i] += max_info_next_time_step

            # Unlearn item
            self.learner.cancel_update()



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
