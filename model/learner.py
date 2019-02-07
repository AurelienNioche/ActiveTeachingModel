import numpy as np

from utils.functions import temporal_difference, softmax
from task import exercise


debug = False


class Learner:

    def __init__(self, alpha=0.1, tau=0.05):

        self.q = np.zeros((exercise.n, exercise.n))
        self.alpha = alpha
        self.tau = tau

    def decide(self, question):

        p = softmax(x=self.q[question, :], temp=self.tau)
        reply = np.random.choice(np.arange(exercise.n), p=p)

        if debug:
            print(f'Question is: {question}')
            print(f'P values are: {[f"{p_i:.02f}" for p_i in p]}')
            print(f'Reply is {reply}')

        return reply

    def learn(self, question, reply, correct_answer):

        success = reply == correct_answer

        old_q_value = self.q[question, reply]
        new_q_value = temporal_difference(v=old_q_value, obs=success, alpha=self.alpha)

        self.q[question, reply] = new_q_value

        if not success:
            self.q[question, correct_answer] = temporal_difference(v=self.q[question, correct_answer], obs=1, alpha=self.alpha)

        if debug:
            print(f'Old q value is {old_q_value}; New q value is {new_q_value}')
