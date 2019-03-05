import numpy as np

from utils.functions import temporal_difference, softmax


debug = False


class Learner:

    def __init__(self, task):
        self.task = task

    def decide(self, question):
        return 0

    def learn(self, question, reply, correct_answer):
        pass

    @property
    def name(self):
        return type(self).__name__


class QLearner(Learner):

    def __init__(self, task, alpha=0.01, tau=0.05):

        super().__init__(task=task)
        self.q = np.zeros((self.task.n, self.task.n))
        self.alpha = alpha
        self.tau = tau

    def decide(self, question):

        p = softmax(x=self.q[question, :], temp=self.tau)
        reply = np.random.choice(np.arange(self.task.n), p=p)

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


class ActRLearner(Learner):

    """
    A chunk is composed of:
    * a type (here: means)
    * several slots (here: slot 1: kanji, slot2: meaning)
    """

    def __init__(self, task, d=0.5, theta=0.0001, s=0.01):

        super().__init__(task=task)

        # Noted 'l' in the paper (to confirm)
        self.n_chunk = task.n

        # Time recording of presentations of chunks
        self.time_presentation = [[] for _ in range(self.n_chunk)]

        # Decay parameter
        self._d = d

        # Retrieval threshold
        self._theta = theta

        # Noise in the activation levels
        self._s = s

        # Time counter
        self.t = 0

        self.square_root_2 = 2**(1/2)

    def activation_function(self, i):

        """The activation of a chunk is the sum of its base-level activation and some noise
        IN FUTURE: ...and the activations it receives from elements attended to. """

        # noise = np.random.normal()
        # print(self.base_level_learning_activation(i))
        return self.base_level_learning_activation(i)  # + noise

    def base_level_learning_activation(self, i):

        """The base-level activation measures how much time has elapsed since the jth use:"""

        # noinspection PyTypeChecker
        sum_a = np.sum([
            (self.t - t_presentation)**(-self._d)
            for t_presentation in self.time_presentation[i]
        ])

        if sum_a > 0:
            return np.log(sum_a)
        else:
            return -np.inf

    def probability_of_retrieval_equation(self, a):

        """The probability of a chunk being above some retrieval threshold τ is"""

        p =  \
            1 / (
                    1 + np.exp
                    (
                        - (a - self._theta) / (self.square_root_2 * self._s)
                    )
                 )
        return p

    def update_time_presentation(self, question):

        # noinspection PyTypeChecker
        self.time_presentation[question].append(self.t)

    def decide(self, question):

        self.t += 1

        a = self.activation_function(question)
        p = self.probability_of_retrieval_equation(a)
        r = np.random.random()
        if p > r:
            reply = question
        else:
            reply = np.random.randint(self.task.n)

        return reply

    def learn(self, question, reply, correct_answer):

        self.update_time_presentation(question)  # We suppose the response to be always correct if recalled


class ActRCogLearner(ActRLearner):

    def __init__(self, task, d=0.5, theta=0.0001, s=0.01, g=0.01, m=0.01):

        super().__init__(task, d, theta, s)

        self.g = g
        self.m = m

    def activation_function(self, i):

        return self.base_level_learning_activation(i) + self.sum_source_activation(i)

    def sum_source_activation(self, i):

        sum_source = 0

        list_j = list(range(self.task.n))
        list_j.remove(i)

        for j in list_j:

            s_j = self.probability_of_retrieval_equation(self.base_level_learning_activation(j))
            assert 0 <= s_j <= 1
            # s_j = self.base_level_learning_activation(j)

            sum_source += \
                self.g * self.task.c_graphic[i, j] * s_j + \
                self.m * self.task.c_semantic[i, j] * s_j

        return sum_source  # / (2*(self.task.n-1))
        # return sum_source
