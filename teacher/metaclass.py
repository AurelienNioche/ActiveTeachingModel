import numpy as np


class GenericTeacher:

    version = 0.0

    def __init__(self, n_item, verbose=False):

        self.n_item = n_item
        self.verbose = verbose

    def ask(self,
            t=None,
            hist_success=None,
            hist_item=None,
            task_param=None,
            student_param=None,
            student_model=None,
            n_item=None,
            n_iteration=None,
            n_possible_replies=None):

        item = \
            self._get_next_node(
                t=t,
                n_iteration=n_iteration,
                hist_success=hist_success,
                hist_item=hist_item,
                task_param=task_param,
                student_param=student_param,
                student_model=student_model)

        if n_possible_replies:
            poss_rep = self._get_possible_replies(
                item, n_item=n_item, n_possible_replies=n_possible_replies)
            return item, poss_rep
        else:
            return item

    def _get_next_node(self, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} is a meta-class."
                                  "This method need to be overridden")

    @staticmethod
    def _get_possible_replies(item, n_item, n_possible_replies):

        # Select randomly possible replies, including the correct one
        all_replies = list(range(n_item))
        all_replies.remove(item)

        possible_replies = \
            [item, ] + list(np.random.choice(
                all_replies, size=n_possible_replies-1, replace=False))
        possible_replies = np.array(possible_replies)
        np.random.shuffle(possible_replies)
        return possible_replies

        # if possible_replies:
        #     poss_rep = self._get_possible_replies(question)
        # else:
        #     poss_rep = None
        #
        # if make_learn:
        #     reply = agent.decide(item=question,
        #                          possible_replies=possible_replies)
        #     agent.learn(item=question)
        #
        #     self.register_question_and_reply(
        #         reply=reply,
        #         question=question,
        #         possible_replies=possible_replies)

    # def register_question_and_reply(self, reply, question,
    #                                 possible_replies=None):
    #
    #     # Update the count of item seen
    #     if self.t > 0:
    #         self.seen[:, self.t] = self.seen[:, self.t - 1]
    #     self.seen[question, self.t] = True
    #
    #     # For backup
    #     self.questions[self.t] = question
    #     self.replies[self.t] = reply
    #     self.successes[self.t] = reply == question
    #     if possible_replies is not None:
    #         self.possible_replies[self.t] = possible_replies
    #
    #     if self.verbose:
    #         print(f"Question chosen: {self.tk.kanji[question]}; "
    #               f"correct answer: {self.tk.meaning[question]}; "
    #               f"possible replies: {self.tk.meaning[possible_replies]};")
    #
    #     self.t += 1

    # def teach(self, agent=None):
    #
    #     tqdm.write("Teaching...")
    #
    #     iterator = tqdm(range(self.tk.n_iteration)) \
    #         if not self.verbose else range(self.tk.n_iteration)
    #
    #     for _ in iterator:
    #         self.ask(agent=agent)
    #
    #     return self.questions, self.replies, self.successes
