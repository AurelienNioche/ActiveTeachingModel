import numpy as np

from utils.functions import temporal_difference, softmax


debug = False


class Learner:

    def __init__(self):
        pass

    def decide(self, question):
        return 0

    def learn(self, question, reply):
        pass

    @property
    def name(self):
        return type(self).__name__


class QLearner(Learner):

    def __init__(self, n_items, alpha=0.01, tau=0.05):

        super().__init__()
        self.n = n_items
        self.q = np.zeros((self.n, self.n))
        self.alpha = alpha
        self.tau = tau

    def decide(self, question):

        p = softmax(x=self.q[question, :], temp=self.tau)
        reply = np.random.choice(np.arange(self.n), p=p)

        if debug:
            print(f'Question is: {question}')
            print(f'P values are: {[f"{p_i:.02f}" for p_i in p]}')
            print(f'Reply is {reply}')

        return reply

    def p_choice(self, question, reply, possible_replies):

        x_i = self.q[question, reply]
        x = self.q[question, possible_replies]

        try:
            p = np.exp(x_i / self.tau) / np.sum(np.exp(x / self.tau))

        except (Warning, FloatingPointError) as w:
            print(x, self.tau)
            raise Exception(f'{w} [x={x}, temp={self.tau}]')

        return p

    def learn(self, question, reply):

        success = question == reply  # We suppose matching (0,0), (1,1) ... (n,n)

        old_q_value = self.q[question, reply]
        new_q_value = temporal_difference(v=old_q_value, obs=success, alpha=self.alpha)

        self.q[question, reply] = new_q_value

        if not success:
            self.q[question, question] = temporal_difference(v=self.q[question, question], obs=1, alpha=self.alpha)

        if debug:
            print(f'Old q value is {old_q_value}; New q value is {new_q_value}')


class ActRLearner(Learner):

    """
    A chunk is composed of:
    * a type (here: means)
    * several slots (here: slot 1: kanji, slot2: meaning)
    """

    def __init__(self, n_items, n_possible_replies=None, d=0.5, tau=0.0001, s=0.01):

        super().__init__()

        self.n_possible_replies = n_possible_replies if n_possible_replies is not None else n_items

        # Number of memory unit
        self.n_chunk = n_items

        # Time recording of presentations of chunks
        self.time_presentation = [[] for _ in range(self.n_chunk)]

        # Decay parameter
        self._d = d

        # Retrieval threshold
        self._tau = tau

        # Noise in the activation levels
        self._s = s

        # Time counter
        self.t = 0

        self.square_root_2 = 2**(1/2)

    def _activation_function(self, i):

        """The activation of a chunk is the sum of its base-level activation and some noise
        IN FUTURE: ...and the activations it receives from elements attended to. """

        # noise = np.random.normal()
        # print(self.base_level_learning_activation(i))
        return self._base_level_learning_activation(i)  # + noise

    def _base_level_learning_activation(self, i):

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

    def _probability_of_retrieval_equation(self, a):

        """The probability of a chunk being above some retrieval threshold τ is"""

        p =  \
            1 / (
                    1 + np.exp
                    (
                        - (a - self._tau) / (self.square_root_2 * self._s)
                    )
                 )
        return p

    def _update_time_presentation(self, question):

        # noinspection PyTypeChecker
        self.time_presentation[question].append(self.t)

    def decide(self, question):

        self.t += 1

        a = self._activation_function(question)
        p = self._probability_of_retrieval_equation(a)
        r = np.random.random()
        if p > r:
            reply = question
        else:
            reply = np.random.randint(self.n_chunk)

        return reply

    def p_choice(self, question, reply):

        self.t += 1

        success = question == reply

        a = self._activation_function(question)
        p = self._probability_of_retrieval_equation(a)

        if success:
            return p
        else:
            return (1-p) / (self.n_possible_replies - 1)

    def learn(self, question, reply):

        self._update_time_presentation(question)  # We suppose the response to be always correct if recalled


class ActRCogLearner(ActRLearner):

    def __init__(self, n_items, c_graphic, c_semantic, d=0.5, theta=0.0001, s=0.01, g=0.01, m=0.01):

        super().__init__(n_items, d, theta, s)

        self.c_graphic = c_graphic
        self.c_semantic = c_semantic

        self.g = g
        self.m = m

    def _activation_function(self, i):

        return self._base_level_learning_activation(i) + self.sum_source_activation(i)

    def sum_source_activation(self, i):

        sum_source = 0

        list_j = list(range(self.n_chunk))
        list_j.remove(i)

        for j in list_j:

            s_j = self._probability_of_retrieval_equation(self._base_level_learning_activation(j))
            assert 0 <= s_j <= 1
            # s_j = self.base_level_learning_activation(j)

            sum_source += \
                self.g * self.c_graphic[i, j] * s_j + \
                self.m * self.c_semantic[i, j] * s_j

        return sum_source  # / (2*(self.task.n-1))
        # return sum_source
