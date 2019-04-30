import numpy as np

from learner.generic import Learner, Task

import task.parameters


class QParam:

    def __init__(self, alpha, tau):
        self.alpha = alpha
        self.tau = tau


class QLearner(Learner):

    def __init__(self, parameters, task_features, verbose=False):

        super().__init__()

        if type(parameters) == dict:
            self.pr = QParam(**parameters)
        elif type(parameters) in (list, tuple, np.ndarray):
            self.pr = QParam(*parameters)
        else:
            raise Exception(f"Type {type(parameters)} is not handled for parameters")

        self.tk = Task(**task_features)

        self.q = np.zeros((self.tk.n_items, self.tk.n_items))
        # print(f"alpha: {self.alpha}, tau:{self.tau}")

        self.verbose = verbose

    def _softmax(self, x):
        try:
            return np.exp(x / self.pr.tau) / np.sum(np.exp(x / self.pr.tau))
        except (Warning, FloatingPointError) as w:
            print(x, self.pr.tau)
            raise Exception(f'{w} [x={x}, temp={self.pr.tau}]')

    def _temporal_difference(self, v, obs=1):

        return v + self.pr.alpha * (obs - v)

    def _softmax_unique(self, x_i, x):

        try:
            p = np.exp(x_i / self.pr.tau) / np.sum(np.exp(x / self.pr.tau))
            return p

        except (Warning, RuntimeWarning, FloatingPointError) as w:
            # print(w, x, self.tau)
            raise Exception(f'{w} [x={x}, temp={self.pr.tau}]')

    def p_recall(self, item):

        x_i = self.q[item, item]
        x = [x_i, ] + [0, ] * (task.parameters.N_POSSIBLE_REPLIES - 1)

        return self._softmax_unique(x_i, x)

    def _p_choice(self, question, reply, possible_replies):

        x_i = self.q[question, reply]
        x = self.q[question, possible_replies]

        return self._softmax_unique(x_i, x)

    def _p_correct(self, question, reply, possible_replies):

        x_correct = self.q[question, question]
        x = self.q[question, possible_replies]

        p_correct = self._softmax_unique(x_correct, x)
        if question == reply:
            return p_correct
        else:
            return 1-p_correct

    def decide(self, question, possible_replies):

        p = self._softmax(x=self.q[question, possible_replies])
        reply = np.random.choice(possible_replies, p=p)

        if self.verbose:
            print(f'Question is: {question}')
            print(f'P values are: {[f"{p_i:.02f}" for p_i in p]}')
            print(f'Reply is {reply}')

        return reply

    def learn(self, question):

        self.q[question, question] = \
            self._temporal_difference(v=self.q[question, question])

        # success = question == reply  # We suppose matching (0,0), (1,1) ... (n,n)
        #
        # old_q_value = self.q[question, reply]
        # new_q_value = temporal_difference(v=old_q_value, obs=success, alpha=self.alpha)
        #
        # self.q[question, reply] = new_q_value
        #
        # if not success:
        #     self.q[question, question] = temporal_difference(v=self.q[question, question], obs=1, alpha=self.alpha)
        #
        # if debug:
        #     print(f'Old q value is {old_q_value}; New q value is {new_q_value}')
